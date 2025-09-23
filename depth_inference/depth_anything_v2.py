import cv2
import glob
import os
import torch
import pandas as pd
import numpy as np

from models.depth_models.Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation


class DepthAnythingV2Inference:
    def __init__(self, input_path, load_from='models/depth_models/Depth_Anything_V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', input_size=518, outdir='results/depth_anything', encoder='vitl', max_depth=80, savenumpy=False, colormap='', eval=False, depth_path='', stream=False):
        self.input_path = input_path
        self.input_size = input_size
        self.outdir = outdir
        self.encoder = encoder
        self.load_from = load_from
        self.max_depth = max_depth
        self.evalualte = eval
        self.depth_path = depth_path
        self.stream = stream
        self.savenumpy=savenumpy
        self.colormap=colormap

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.depth_anything = DepthAnythingV2(**{**self.model_configs[self.encoder], 'max_depth': self.max_depth})
        self.depth_anything.load_state_dict(torch.load(self.load_from, map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.DEVICE).eval()

        # self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/test.pt')
        self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/yolov8n.pt')
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)

        self.process_files()


    def process_files(self):
        if os.path.isfile(self.input_path):
            self.filenames = [self.input_path]
        else:
            self.filenames = glob.glob(os.path.join(self.input_path, '**/*'), recursive=True)
        
        os.makedirs(self.outdir, exist_ok=True)

    def process_images(self):
        for k, filename in enumerate(self.filenames):
            print(f'Progress {k+1}/{len(self.filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            depth = self.depth_anything.infer_image(raw_image, self.input_size)
            output_path = os.path.join(self.outdir,'frame_{}.png'.format(k))

            predictions = self.model.predict(raw_image)

            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_image, predictions)
            if self.evalualte:
                depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
            cv2.imwrite(output_path, depth_frame)
            
            
            if self.savenumpy:
                    output_path = os.path.join(self.savenumpy, os.path.splitext(os.path.basename(filename))[0] + '_numpy_matrix' + '_raw_depth_meter_frame.npy')
                    np.save(output_path, depth)

            if self.colormap:
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                depth = (self.cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                output_path = os.path.join(self.colormap, os.path.splitext(os.path.basename(filename))[0] + '_colormap_' + 'frame.png')
                
                cv2.imwrite(output_path, depth)
        
        
            if self.savenumpy:
                    output_path = os.path.join(self.savenumpy, os.path.splitext(os.path.basename(filename))[0] + '_numpy_matrix' + '_raw_depth_meter_frame.npy')
                    np.save(output_path, depth)

            if self.colormap:
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                depth = (self.cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                output_path = os.path.join(self.colormap, os.path.splitext(os.path.basename(filename))[0] + '_colormap_' + 'frame.png')
                
                cv2.imwrite(output_path, depth)


        
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
                
                depth = self.depth_anything.infer_image(raw_frame, self.input_size)

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
            depth = self.depth_anything.infer_image(raw_frame, self.input_size)
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
            if(k+1==20):
                break
        df = pd.DataFrame(data, columns=['Frame','UTC_time','x1','y1','x2','y2','CLASS', 'predicted_depth', 'actual_depth'])
        df.to_csv('{}/result.csv'.format(self.outdir), index=False)
        # out.release()
        # out.release()
        print(f'Output saved to {self.outdir}')
        # out.release()
        # print(f'Output saved to {self.outdir}')