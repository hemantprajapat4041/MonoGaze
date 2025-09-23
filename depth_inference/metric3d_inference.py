import os
import sys

# 1. Directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ascend two levels to project root (MonoGaze/)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 3. Insert project root so that models/ and utils/ become importable
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. Also insert the Metric3D repo folder directly
metric3d_root = os.path.join(project_root, 'models', 'depth_models', 'Metric3D')
if metric3d_root not in sys.path:
    sys.path.insert(0, metric3d_root)


import cv2
from PIL import Image
import glob
import os
import torch
import pandas as pd
import numpy as np
import math
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation

class Metric3DInference:
    def __init__(self, input_path, model_path='', outdir='results/metric3d', 
                 max_depth=80, savenumpy=False, colormap='', eval=False, 
                 depth_path='', stream=False, model_type='mono'):
        
        print(f"Initializing Metric3DInference with input_path: {input_path}")
        self.input_path = input_path
        self.outdir = outdir
        self.model_path = model_path
        self.max_depth = max_depth
        self.evalualte = eval
        self.depth_path = depth_path
        self.stream = stream
        self.savenumpy = savenumpy
        self.colormap = colormap
        self.model_type = model_type
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.DEVICE}")
        
        # Load Metric3D model
        print("Loading Metric3D model...")
        self.depth_model = self.load_metric3d_model()
        
        # YOLO and depth approximation setup
        print("Setting up YOLO and depth approximation...")
        self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/yolov8n.pt')
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)
        
        print("Processing files...")
        self.process_files()
        print(f"Found {len(self.filenames)} files to process")
    
    def load_metric3d_model(self):
        """
        Load Metric3D model
        """
        try:
            # Import Metric3D
            import sys
            metric3d_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_models', 'Metric3D')
            if metric3d_path not in sys.path:
                sys.path.insert(0, metric3d_path)
            
            # Metric3D uses hubconf for model loading
            import hubconf
            
            # Load the model using hubconf
            if self.model_type == 'mono':
                depth_model = hubconf.metric3d_convnext_tiny(pretrained=True)
            else:
                depth_model = hubconf.metric3d_convnext_tiny(pretrained=True)
            
            depth_model = depth_model.to(self.DEVICE).eval()
            
            print(f"Loaded Metric3D model successfully")
            return depth_model
            
        except ImportError as e:
            print(f"Error importing Metric3D: {e}")
            print("Make sure the Metric3D repository is properly cloned and accessible")
            return None
        except Exception as e:
            print(f"Error loading Metric3D model: {e}")
            return None
    
    def preprocess_image_for_metric3d(self, raw_image):
        """
        Metric3D preprocessing:
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
        
        # Metric3D typically works with 640x480 or similar dimensions
        target_width, target_height = 640, 480
        
        # Resize image maintaining aspect ratio
        aspect_ratio = original_width / original_height
        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        resized_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Pad to target dimensions if necessary
        if new_width != target_width or new_height != target_height:
            padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
            resized_image = padded_image
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(resized_image).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        return image_tensor, (original_height, original_width), (target_height, target_width)
    
    def infer_depth(self, raw_image):
        """
        Metric3D depth inference
        """
        if self.depth_model is None:
            print("Error: Metric3D model not loaded")
            return None
        
        with torch.no_grad():
            # Preprocess for Metric3D
            image_tensor, original_size, target_size = self.preprocess_image_for_metric3d(raw_image)
            image_tensor = image_tensor.to(self.DEVICE)
            
            # Create camera model (simplified - using default values)
            # Metric3D expects camera intrinsic parameters
            h, w = target_size
            fx = fy = max(h, w)  # Simplified focal length
            cx, cy = w // 2, h // 2  # Principal point at center
            
            # Create camera model using the same format as Metric3D
            intrinsic = [fx, fy, cx, cy]
            cam_model = self.build_camera_model(h, w, intrinsic)
            cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
            cam_model = cam_model[None, :, :, :].to(self.DEVICE)
            
            # Create camera model stacks as expected by Metric3D
            cam_model_stacks = [
                torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
                for i in [2, 4, 8, 16, 32]
            ]
            
            # Prepare data dictionary as expected by Metric3D
            data = {
                'input': image_tensor,
                'cam_model': cam_model_stacks
            }
            
            # Metric3D inference
            pred_depth, confidence, output_dict = self.depth_model.inference(data)
            
            # Convert to numpy array
            if hasattr(pred_depth, 'cpu'):
                pred_depth = pred_depth.cpu().numpy()
            elif hasattr(pred_depth, 'numpy'):
                pred_depth = pred_depth.numpy()
            
            # Ensure depth is 2D
            if len(pred_depth.shape) > 2:
                pred_depth = pred_depth.squeeze()
            
            # Resize depth back to original image dimensions
            depth_resized = cv2.resize(pred_depth, (original_size[1], original_size[0]))
            
            return depth_resized
    
    def build_camera_model(self, H: int, W: int, intrinsics: list) -> np.array:
        """
        Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map.
        """
        fx, fy, u0, v0 = intrinsics
        f = (fx + fy) / 2.0
        # principle point location
        x_row = np.arange(0, W).astype(np.float32)
        x_row_center_norm = (x_row - u0) / W
        x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

        y_col = np.arange(0, H).astype(np.float32) 
        y_col_center_norm = (y_col - v0) / H
        y_center = np.tile(y_col_center_norm, (W, 1)).T # [H, W]

        # FoV
        fov_x = np.arctan(x_center / (f / W))
        fov_y = np.arctan(y_center / (f / H))

        cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
        return cam_model
    
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
            
            # Metric3D depth inference
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
