import os
import sys

# 1. Directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ascend two levels to project root (MonoGaze/)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 3. Insert project root so that models/ and utils/ become importable
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. Also insert the HRDepth repo folder directly
hrdepth_root = os.path.join(project_root, 'models', 'depth_models', 'HRDepth')
if hrdepth_root not in sys.path:
    sys.path.insert(0, hrdepth_root)


import cv2
from PIL import Image
import glob
import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation

class HRDepthInference:
    def __init__(self, input_path, model_path='', outdir='results/hrdepth', 
                 max_depth=80, savenumpy=False, colormap='', eval=False, 
                 depth_path='', stream=False, encoder_type='resnet18'):
        
        self.input_path = input_path
        self.outdir = outdir
        self.model_path = model_path
        self.max_depth = max_depth
        self.evalualte = eval
        self.depth_path = depth_path
        self.stream = stream
        self.savenumpy = savenumpy
        self.colormap = colormap
        self.encoder_type = encoder_type
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Load HRDepth model
        self.depth_encoder, self.depth_decoder = self.load_hrdepth_model()
        
        # YOLO and depth approximation setup
        self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/yolov8n.pt')
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)
        
        self.process_files()
    
    def load_hrdepth_model(self):
        """
        Load HRDepth encoder and decoder models
        """
        try:
            # Import HRDepth networks
            import sys
            hrdepth_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_models', 'HRDepth')
            if hrdepth_path not in sys.path:
                sys.path.insert(0, hrdepth_path)
            
            from models.depth_models.HRDepth.networks import ResnetEncoder, HRDepthDecoder
            
            # Initialize encoder (default to ResNet18)
            if self.encoder_type == 'resnet18':
                depth_encoder = ResnetEncoder(18, False)
            elif self.encoder_type == 'resnet50':
                depth_encoder = ResnetEncoder(50, False)
            else:
                depth_encoder = ResnetEncoder(18, False)
            
            # Initialize HRDepth decoder
            depth_decoder = HRDepthDecoder(depth_encoder.num_ch_enc)
            
            # Load pretrained weights if provided
            if self.model_path and os.path.exists(self.model_path):
                if os.path.isdir(self.model_path):
                    # Look for encoder.pth and depth.pth in the directory
                    encoder_path = os.path.join(self.model_path, 'encoder.pth')
                    decoder_path = os.path.join(self.model_path, 'depth.pth')
                    
                    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
                        encoder_dict = torch.load(encoder_path, map_location=self.DEVICE)
                        decoder_dict = torch.load(decoder_path, map_location=self.DEVICE)
                        
                        # Load encoder weights
                        load_dict = {k: v for k, v in encoder_dict.items() if k in depth_encoder.state_dict()}
                        depth_encoder.load_state_dict(load_dict)
                        
                        # Load decoder weights
                        depth_decoder.load_state_dict(decoder_dict)
                        
                        print(f"Loaded HRDepth model from {self.model_path}")
                    else:
                        print(f"Warning: encoder.pth or depth.pth not found in {self.model_path}")
                else:
                    print(f"Warning: Model path {self.model_path} is not a directory")
            
            # Move models to device and set to eval mode
            depth_encoder = depth_encoder.to(self.DEVICE).eval()
            depth_decoder = depth_decoder.to(self.DEVICE).eval()
            
            return depth_encoder, depth_decoder
            
        except ImportError as e:
            print(f"Error importing HRDepth networks: {e}")
            print("Make sure the HRDepth repository is properly cloned and accessible")
            return None, None
        except Exception as e:
            print(f"Error loading HRDepth model: {e}")
            return None, None
    
    def preprocess_image_for_hrdepth(self, raw_image):
        """
        HRDepth preprocessing:
        - Convert BGR to RGB
        - Resize to model input dimensions
        - Normalize with ImageNet stats
        """
        # Convert BGR to RGB
        if len(raw_image.shape) == 3 and raw_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = raw_image
        
        # Get original dimensions
        original_height, original_width = rgb_image.shape[:2]
        
        # Resize to model input dimensions (default to 1280x384 as per HRDepth paper)
        target_width, target_height = 1280, 384
        
        # Resize image
        resized_image = cv2.resize(rgb_image, (target_width, target_height))
        
        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(resized_image).unsqueeze(0)
        
        return image_tensor, (original_height, original_width), (target_height, target_width)
    
    def infer_depth(self, raw_image):
        """
        HRDepth depth inference
        """
        if self.depth_encoder is None or self.depth_decoder is None:
            print("Error: HRDepth model not loaded")
            return None
        
        with torch.no_grad():
            # Preprocess for HRDepth
            image_tensor, original_size, target_size = self.preprocess_image_for_hrdepth(raw_image)
            image_tensor = image_tensor.to(self.DEVICE)
            
            # Forward pass through encoder and decoder
            features = self.depth_encoder(image_tensor)
            outputs = self.depth_decoder(features)
            
            # Extract disparity from outputs
            if ("disparity", "Scale0") in outputs:
                disparity = outputs[("disparity", "Scale0")].cpu().detach().squeeze().numpy()
            else:
                # Fallback: try to get the first output
                first_output = list(outputs.values())[0]
                disparity = first_output.cpu().detach().squeeze().numpy()
            
            # Convert disparity to depth (disparity = 1/depth)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            depth = 1.0 / (disparity + epsilon)
            
            # Resize depth back to original image dimensions
            depth_resized = cv2.resize(depth, (original_size[1], original_size[0]))
            
            return depth_resized
    
    def process_files(self):
        if os.path.isfile(self.input_path):
            self.filenames = [self.input_path]
        else:
            self.filenames = glob.glob(os.path.join(self.input_path, '/*'), recursive=True)
        os.makedirs(self.outdir, exist_ok=True)
    
    def process_images(self):
        for k, filename in enumerate(self.filenames):
            print(f'Progress {k+1}/{len(self.filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            # HRDepth depth inference
            depth = self.infer_depth(raw_image)
            
            if depth is None:
                print(f"Failed to process {filename}")
                continue
            
            output_path = os.path.join(self.outdir,'frame_{}.png'.format(k))
            
            # YOLO predictions and depth extraction
            predictions = self.model.predict(raw_image)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_image, predictions)
            
            if self.evalualte:
                depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
            
            cv2.imwrite(output_path, depth_frame)
            
        print(f'Output saved to {self.outdir}')

    def process_video(self, fps=30):
        for k, filename in enumerate(self.filenames):
            raw_video = cv2.VideoCapture(filename)
            length = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))/3
            output_path = os.path.join(self.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
            frame=1
            print(f'Progress {k+1}/{len(self.filenames)}: {filename}')
            while raw_video.isOpened():
                ret, raw_frame = raw_video.read()
                if not ret:
                    break
                
                depth = self.infer_depth(raw_frame)
                
                if depth is None:
                    print(f"Failed to process frame {frame}")
                    frame += 1
                    continue

                predictions = self.model.predict(raw_frame)
                predictions = self.depth_approximator.depth(predictions, depth)
                depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)
                if self.evalualte:
                    depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
                out.write(depth_frame)
                print(f'Frame: {frame}/{length} complete')
                frame+=1
            
            raw_video.release()
            out.release()
        print(f'Output saved to {self.outdir}')

    def eval_stereo(self, fps=4):
        data = []
        df = pd.read_csv(r"F:\data\timestamps.csv")
        for k, filename in enumerate(self.filenames):
            file_name = os.path.join(self.input_path, 'frame_{}.jpg'.format(k+4201))
            print(f'Progress {k+1}/{len(self.filenames)}: {file_name}')
            raw_frame = cv2.imread(file_name)
            depth = self.infer_depth(raw_frame)
            
            if depth is None:
                print(f"Failed to process {file_name}")
                continue
                
            output_path = os.path.join(self.outdir,'frame_{}.png'.format(k+4201))
            predictions = self.model.predict(raw_frame)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)
            depth_frame, predictions = self.depth_approximator.evaluate(self.depth_path, depth_frame, file_name, predictions)
            cv2.imwrite(output_path,depth_frame)
            for pred in predictions:
                res = [df[df['frame_number']==k+4201]['frame_number'].to_string()[5:].strip(),df[df['frame_number']==k+4201]['utc_timestamp'].to_string()[5:].strip(),pred['x1'],pred['y1'],pred['x2'],pred['y2'], pred['class'], pred['estimated_depth'], pred['actual_depth']]               
                data.append(res)
            if(k+1==5):
                break
        df = pd.DataFrame(data, columns=['Frame','UTC_time','x1','y1','x2','y2','CLASS', 'predicted_depth', 'actual_depth'])
        df.to_csv('{}/result.csv'.format(self.outdir), index=False)
        print(f'Output saved to {self.outdir}')
