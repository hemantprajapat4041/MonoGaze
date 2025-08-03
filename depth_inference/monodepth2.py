from __future__ import absolute_import, division, print_function

import os
import glob
import cv2
import PIL.Image as pil
import numpy as np

import torch
from torchvision import transforms

import models.depth_models.Mono_Depth_2.networks as networks
from models.depth_models.Mono_Depth_2.layers import disp_to_depth
from models.depth_models.Mono_Depth_2.evaluate_depth import STEREO_SCALE_FACTOR
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation

class MonoDepth2Inference:
    def __init__(self, input_path, outdir='results/monodepth2', stream=False):
        self.input_path = input_path
        self.outdir = outdir
        self.stream = stream
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pretrained = int(input('Enter 1 to use pretrained model on KITTI dataset, 0 non pretrained model:'))

        if self.pretrained:
            self.model_path = 'models/depth_models/Mono_Depth_2/models/mono+stereo_1024x320'
        else:
            self.model_path = 'models/depth_models/Mono_Depth_2/models/mono+stereo_no_pt_640x192'

        self.encoder_path = os.path.join(self.model_path, 'encoder.pth')
        self.depth_decoder_path = os.path.join(self.model_path, 'depth.pth')
        
        self.encoder = networks.ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)
        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(self.depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

        self.detection_model = Yolo2DObjectDetection(model_path='models/detection_models/bounding_box/test.pt')
        self.depth_approximator = DepthApproximation()

    def process_files(self):
        if os.path.isfile(self.input_path):
            paths = [self.input_path]
        elif os.path.isdir(self.input_path):
            # Searching folder for images
            paths = glob.glob(os.path.join(self.input_path, '*.{}'.format('jpg')))
        else:
            raise Exception("Can not find args.image_path: {}".format(self.input_path))

        os.makedirs(self.outdir, exist_ok=True)
        return paths

    def process_image(self):
        paths = self.process_files()
        with torch.no_grad():
            for idx, image_path in enumerate(paths):
                print(f'Progress {idx+1}/{len(paths)}: {image_path}')
                # Load image and preprocess
                input_image = pil.open(image_path).convert('RGB')
                original_width, original_height = input_image.size
                input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # PREDICTION
                input_image = input_image.to(self.device)
                features = self.encoder(input_image)
                outputs = self.depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                output_name = os.path.splitext(os.path.basename(image_path))[0]
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                name_dest = os.path.join(self.outdir, "{}_depth_{}.jpg".format(output_name, self.pretrained))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy().squeeze()
                
                img = cv2.imread(image_path)
                predictions = self.detection_model.predict(img)
                predictions = self.detection_model.lanczos_conversion(predictions, (original_width, original_height), (self.feed_width, self.feed_height))
                predictions = self.depth_approximator.depth(predictions, metric_depth, False, False)
                predictions = self.detection_model.lanczos_conversion(predictions, (self.feed_width, self.feed_height), (original_width, original_height))
                depth_frame = self.depth_approximator.annotate_depth_on_img(img, predictions)
                cv2.imwrite(name_dest, depth_frame)


        print('Output saved to {}'.format(self.outdir))

    def process_video(self, fps=30):
        paths = self.process_files()
        frame=1
        with torch.no_grad():
            if not self.stream:
                for k, filename in enumerate(paths):
                    raw_video = cv2.VideoCapture(filename)
                    length = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))/3
                    output_name = os.path.splitext(os.path.basename(filename))[0]
                    name_dest = os.path.join(self.outdir, "{}_depth_{}.mp4".format(output_name, self.pretrained))
                    out = cv2.VideoWriter(name_dest, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
                    print(f'Progress {k+1}/{len(paths)}: {filename}')
                    while raw_video.isOpened():
                        ret, raw_frame = raw_video.read()
                        if not ret:
                            break
                        original_width, original_height = raw_frame.shape[:2]
                        input_image = pil.fromarray(raw_frame).convert('RGB')
                        input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
                        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                        # PREDICTION
                        input_image = input_image.to(self.device)
                        features = self.encoder(input_image)
                        outputs = self.depth_decoder(features)

                        disp = outputs[("disp", 0)]
                        disp_resized = torch.nn.functional.interpolate(
                            disp, (original_height, original_width), mode="bilinear", align_corners=False)

                        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy().squeeze()
                        
                        predictions = self.detection_model.predict(raw_frame)
                        predictions = self.detection_model.lanczos_conversion(predictions, (original_width, original_height), (self.feed_width, self.feed_height))
                        predictions = self.depth_approximator.depth(predictions, metric_depth, False, False)
                        predictions = self.detection_model.lanczos_conversion(predictions, (self.feed_width, self.feed_height), (original_width, original_height))
                        depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)
                        out.write(depth_frame)
                        print(f'Frame: {frame}/{length} complete')
                        frame+=1
                    
                    raw_video.release()
                    out.release()
            else:
                self.filenames.sort()
                for k, filename in enumerate(paths):
                    output_name = os.path.splitext(os.path.basename(filename))[0]
                    name_dest = os.path.join(self.outdir, "{}_depth_{}.mp4".format(output_name, self.pretrained))
                    print(f'Progress {k+1}/{len(self.filenames)}: {filename}')
                    raw_frame = cv2.imread(filename)
                    img_width, img_height = raw_frame.shape[:2]
                    out = cv2.VideoWriter(name_dest, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img_width, img_height))
                    predictions = self.model.predict(raw_frame)

                    predictions = self.depth_approximator.depth(predictions, depth)
                    depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)
                    if self.evalualte:
                        depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
                    out.write(depth_frame)
                out.release()
            print(f'Output saved to {self.outdir}')
