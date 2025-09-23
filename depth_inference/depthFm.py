import os
import sys

# 1. Directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ascend two levels to project root (MonoGaze/)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 3. Insert project root so that models/ and utils/ become importable
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. Also insert the DepthFM repo folder directly
depthfm_root = os.path.join(project_root, 'models', 'depth_models', 'DepthFM')
if depthfm_root not in sys.path:
    sys.path.insert(0, depthfm_root)


import cv2
from PIL import Image
import glob
import os
import torch
import pandas as pd
import numpy as np
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation

class DepthFMInference:
    def __init__(self, input_path, model_path='', outdir='results/depthfm', 
                 max_depth=80, savenumpy=False, colormap='', eval=False, 
                 depth_path='', stream=False, num_steps=2, ensemble_size=4):
        
        self.input_path = input_path
        self.outdir = outdir
        self.model_path = model_path
        self.max_depth = max_depth
        self.evalualte = eval
        self.depth_path = depth_path
        self.stream = stream
        self.savenumpy = savenumpy
        self.colormap = colormap
        self.num_steps = num_steps
        self.ensemble_size = ensemble_size
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Load DepthFM model
        self.depth_model = self.load_depthfm_model()
        
        # YOLO and depth approximation setup
        self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/yolov8n.pt')
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)
        
        self.process_files()
    
    def load_depthfm_model(self):
        """
        Load DepthFM model
        """
        try:
            # Import DepthFM
            import sys
            depthfm_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_models', 'DepthFM')
            if depthfm_path not in sys.path:
                sys.path.insert(0, depthfm_path)
            
            from models.depth_models.DepthFM.depthfm import DepthFM
            
            # Initialize DepthFM model
            checkpoint_path = self.model_path if self.model_path else os.path.join(depthfm_path, "checkpoints/depthfm-v1.ckpt")
            depth_model = DepthFM(checkpoint_path)
            depth_model = depth_model.to(self.DEVICE).eval()
            
            print(f"Loaded DepthFM model successfully")
            return depth_model
            
        except ImportError as e:
            print(f"Error importing DepthFM: {e}")
            print("Make sure the DepthFM repository is properly cloned and accessible")
            return None
        except Exception as e:
            print(f"Error loading DepthFM model: {e}")
            return None
    
    def preprocess_image_for_depthfm(self, raw_image):
        """
        DepthFM preprocessing:
        - Convert BGR to RGB
        - Resize to model input dimensions
        - Normalize appropriately
        """
        # Convert BGR to RGB
        if len(raw_image.shape) == 3 and raw_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = raw_image
        
        # Get original dimensions
        original_height, original_width = rgb_image.shape[:2]
        
        # DepthFM typically works with 512x512 square dimensions
        target_size = 512
        
        # Resize image maintaining aspect ratio
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        
        resized_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Pad to square if necessary
        if new_width != target_size or new_height != target_size:
            padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - new_height) // 2
            x_offset = (target_size - new_width) // 2
            padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
            resized_image = padded_image
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(resized_image).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        return image_tensor, (original_height, original_width), (target_size, target_size)
    
    def infer_depth(self, raw_image):
        """
        DepthFM depth inference
        """
        if self.depth_model is None:
            print("Error: DepthFM model not loaded")
            return None
        
        with torch.no_grad():
            # Preprocess for DepthFM
            image_tensor, original_size, target_size = self.preprocess_image_for_depthfm(raw_image)
            image_tensor = image_tensor.to(self.DEVICE)
            
            # DepthFM inference
            depth = self.depth_model(image_tensor, num_steps=self.num_steps, ensemble_size=self.ensemble_size)
            
            # Convert to numpy array
            if hasattr(depth, 'cpu'):
                depth = depth.cpu().numpy()
            elif hasattr(depth, 'numpy'):
                depth = depth.numpy()
            
            # Ensure depth is 2D
            if len(depth.shape) > 2:
                depth = depth.squeeze()
            
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
            
            # DepthFM depth inference
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