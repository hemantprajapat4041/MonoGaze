import os
import sys

# 1. Directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ascend two levels to project root (MonoGaze/)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 3. Insert project root so that models/ and utils/ become importable
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. Also insert the ZoeDepth repo folder directly
zoedepth_root = os.path.join(project_root, 'models', 'depth_models', 'ZoeDepth')
if zoedepth_root not in sys.path:
    sys.path.insert(0, zoedepth_root)


import cv2
from PIL import Image
import glob
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation

class ZeoDepthInference:
    def __init__(self, input_path, model_name='Intel/zoedepth-nyu-kitti', 
                 outdir='results/zoedepth', max_depth=80, savenumpy=False, 
                 colormap='', eval=False, depth_path='', stream=False):
        
        self.input_path = input_path
        self.outdir = outdir
        self.model_name = model_name
        self.max_depth = max_depth
        self.evalualte = eval
        self.depth_path = depth_path
        self.stream = stream
        self.savenumpy = savenumpy
        self.colormap = colormap
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Load ZoeDepth model with automatic preprocessing
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.depth_model = ZoeDepthForDepthEstimation.from_pretrained(self.model_name)
        self.depth_model = self.depth_model.to(self.DEVICE).eval()
        
        # YOLO and depth approximation setup
        self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/yolov8n.pt')
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)
        
        self.process_files()
    
    def preprocess_image_for_zoedepth(self, raw_image):
        """
        ZoeDepth preprocessing:
        - Handles ImageNet normalization automatically
        - Pads to multiple of 32
        - Domain classification for appropriate head selection
        """
        # Convert BGR to RGB
        if len(raw_image.shape) == 3 and raw_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = raw_image
            
        from PIL import Image
        pil_image = Image.fromarray(rgb_image.astype(np.uint8))
        
        return pil_image
    
    def infer_depth(self, raw_image):
        """
        ZoeDepth depth inference with ensemble and domain adaptation
        """
        with torch.no_grad():
            # Preprocess for ZoeDepth
            pil_image = self.preprocess_image_for_zoedepth(raw_image)
            original_size = (pil_image.height, pil_image.width)
            
            # ZoeDepth automatic preprocessing with domain selection
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}
            
            # Forward pass with ensemble (ZoeDepth's standard practice)
            outputs = self.depth_model(**inputs)
            
            # Ensemble with horizontally flipped version for better accuracy
            inputs_flipped = self.processor(
                images=pil_image.transpose(Image.FLIP_LEFT_RIGHT), 
                return_tensors="pt"
            )
            inputs_flipped = {k: v.to(self.DEVICE) for k, v in inputs_flipped.items()}
            outputs_flipped = self.depth_model(**inputs_flipped)
            
            # Post-process with ensemble
            post_processed = self.processor.post_process_depth_estimation(
                outputs, 
                source_sizes=[original_size],
                outputs_flipped=outputs_flipped
            )
            
            depth = post_processed[0]["predicted_depth"].cpu().numpy()
            return depth  # Metric depth in meters
    
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
            
            # ZoeDepth depth inference
            depth = self.infer_depth(raw_image)
            
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
        # output_path = os.path.join(self.outdir,'video.mp4')
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
        data = []
        df = pd.read_csv(r"F:\data\timestamps.csv")
        for k, filename in enumerate(self.filenames):
            file_name = os.path.join(self.input_path, 'frame_{}.jpg'.format(k+4201))
            print(f'Progress {k+1}/{len(self.filenames)}: {file_name}')
            raw_frame = cv2.imread(file_name)
            depth = self.infer_depth(raw_frame)
            output_path = os.path.join(self.outdir,'frame_{}.png'.format(k+4201)) ## new change for getting output as annoted images instead of video file
            predictions = self.model.predict(raw_frame)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)
            depth_frame, predictions = self.depth_approximator.evaluate(self.depth_path, depth_frame, file_name, predictions)
            # out.write(depth_frame)
            cv2.imwrite(output_path,depth_frame)
            for pred in predictions:
                res = [df[df['frame_number']==k+4201]['frame_number'].to_string()[5:].strip(),df[df['frame_number']==k+4201]['utc_timestamp'].to_string()[5:].strip(),pred['x1'],pred['y1'],pred['x2'],pred['y2'], pred['class'], pred['estimated_depth'], pred['actual_depth']]               
                data.append(res)
            if(k+1==5):
                break
        df = pd.DataFrame(data, columns=['Frame','UTC_time','x1','y1','x2','y2','CLASS', 'predicted_depth', 'actual_depth'])
        df.to_csv('{}/result.csv'.format(self.outdir), index=False)
        # out.release()
        # out.release()
        print(f'Output saved to {self.outdir}')
        # out.release()
        # print(f'Output saved to {self.outdir}')