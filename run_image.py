import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from models.depth_models.Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from utils.yolo.functions import Yolo2DObjectDetection
from utils.generic_functions.functions import DepthApproximation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./results/image')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='models/depth_models/Depth_Anything_V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth')
    parser.add_argument('--max-depth', type=float,dest='max_depth', default=80)
    
    parser.add_argument('--savenumpy', type=str, help='save the model raw output')
    parser.add_argument('--colormap', type=str, default='', help='only display the prediction')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.input_path):
        if args.input_path.endswith('txt'):
            with open(args.input_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.input_path]
    else:
        filenames = glob.glob(os.path.join(args.input_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.colormap:
        os.makedirs(args.colormap, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    model = Yolo2DObjectDetection('models/detection_models/bounding_box/test.pt')
    depth_approximator = DepthApproximation(max_depth=args.max_depth)
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_frame.png')

        predictions = model.predict(raw_image)

        predictions = depth_approximator.depth(predictions, depth)
        depth_frame = depth_approximator.annotate_depth_on_img(raw_image, predictions)
        cv2.imwrite(output_path, depth_frame)
        
        if args.savenumpy:
                output_path = os.path.join(args.savenumpy, os.path.splitext(os.path.basename(filename))[0] + '_numpy_matrix' + '_raw_depth_meter_frame.npy')
                np.save(output_path, depth)

        if args.colormap:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            output_path = os.path.join(args.colormap, os.path.splitext(os.path.basename(filename))[0] + '_colormap_' + 'frame.png')
            
            cv2.imwrite(output_path, depth)