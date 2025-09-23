import os
import sys

import diffusers

# 1. Directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ascend two levels to project root (MonoGaze/)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 3. Insert project root so that models/ and utils/ become importable
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. Also insert the Marigold repo folder directly
marigold_root = os.path.join(project_root, 'models', 'depth_models', 'Marigold')
if marigold_root not in sys.path:
    sys.path.insert(0, marigold_root)


import cv2
from PIL import Image
import glob
import os
import torch
import pandas as pd
import numpy as np
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation

class MarigoldInference:
    def __init__(self, input_path, model_path='', outdir='results/marigold', 
                 max_depth=80, savenumpy=False, colormap='', eval=False, 
                 depth_path='', stream=False, model_variant='v1_4'):
        
        print(f"Initializing MarigoldInference with input_path: {input_path}")
        self.input_path = input_path
        self.outdir = outdir
        self.model_path = model_path
        self.max_depth = max_depth
        self.evalualte = eval
        self.depth_path = depth_path
        self.stream = stream
        self.savenumpy = savenumpy
        self.colormap = colormap
        self.model_variant = model_variant
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.DEVICE}")
        
        # Load Marigold model
        print("Loading Marigold model...")
        self.depth_model = self.load_marigold_model()
        
        # YOLO and depth approximation setup
        print("Setting up YOLO and depth approximation...")
        self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/yolov8n.pt')
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)
        
        print("Processing files...")
        self.process_files()
        print(f"Found {len(self.filenames)} files to process")
    
    def load_marigold_model(self):
        """
        Load Marigold model
        """
        try:
            # Import Marigold
            import sys
            marigold_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_models', 'Marigold')
            if marigold_path not in sys.path:
                sys.path.insert(0, marigold_path)
            
            from models.depth_models.Marigold.marigold.marigold_depth_pipeline import MarigoldDepthPipeline
            
            # Initialize Marigold model using local pipeline
            depth_model = MarigoldDepthPipeline.from_pretrained(
                "prs-eth/marigold-depth-v1-1",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            depth_model = depth_model.to(self.DEVICE)
            print(f"Loaded Marigold model successfully")
            return depth_model
        except ImportError as e:
            print(f"Error importing Marigold: {e}")
            print("Make sure the Marigold repository is properly cloned and accessible")
            return None
        except Exception as e:
            print(f"Error loading Marigold model: {e}")
            return None
    
    def preprocess_image_for_marigold(self, raw_image):
        """
        Marigold preprocessing:
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
        
        # Marigold typically works with 768x768 or similar square dimensions
        target_size = 768
        
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
        
        # Convert to PIL Image for Marigold
        pil_image = Image.fromarray(resized_image)
        
        return pil_image, (original_height, original_width), (target_size, target_size)
    
    def infer_depth(self, raw_image):
        """
        Marigold depth inference
        """
        if self.depth_model is None:
            print("Error: Marigold model not loaded")
            return None
        
        with torch.no_grad():
            # Preprocess for Marigold
            pil_image, original_size, target_size = self.preprocess_image_for_marigold(raw_image)
            
            # Marigold inference
            result = self.depth_model(
                pil_image,
                output_type="np",
                denoising_steps=10,
                ensemble_size=5
            )
            
            # Extract depth from result
            depth = result.depth_np
            
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
            if raw_image is None:
                print(f"Failed to load image: {filename}")
                continue
            
            print(f"Processing image with shape: {raw_image.shape}")
            
            # Marigold depth inference
            depth = self.infer_depth(raw_image)
            
            if depth is None:
                print(f"Failed to process {filename}")
                continue
            
            print(f"Depth inference successful, depth shape: {depth.shape}")
            
            output_path = os.path.join(self.outdir,'frame_{}.png'.format(k))
            print(f"Saving to: {output_path}")
            
            # YOLO predictions and depth extraction
            predictions = self.model.predict(raw_image)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_image, predictions)
            
            if self.evalualte:
                depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
            
            cv2.imwrite(output_path, depth_frame)
            print(f"Saved depth frame to: {output_path}")
            
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