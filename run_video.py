import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from  Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

from inference_sdk import InferenceHTTPClient

def decode_depth(val):
    NewValue = val*80/255
    return NewValue

def get_object_depth(predictions, depth_grid,img):
    pts = []
    for pred in predictions:
        pts.append([[int(pred['x']-pred['width']/2), int(pred['y']-pred['height']/2)],[int(pred['x']+pred['width']/2), int(pred['y']+pred['height']/2)]])  
    
    depth_all =[]
    for i,j in pts:
        count=0
        depth=0
        for x in range(i[0],j[0]+1):
            for y in range(i[1],j[1]+1):
                depth+=decode_depth(depth_grid[y][x])
                count+=1
        depth=depth/count
        depth_all.append(depth)

    index=0
    for pred in predictions:
        pred.update({'estimated_depth':depth_all[index]})
        index+=1
        cv2.rectangle(img=img, pt1=(int(pred['x']-pred['width']/2), int(pred['y']-pred['height']/2)),
                            pt2=(int(pred['x']+pred['width']/2), int(pred['y']+pred['height']/2)),
                            color=(0,255,255), thickness=3)# cv2.imshow('Result', img)
        pred.update({'detection_id': index})
        depth=str(pred['estimated_depth'])
        img = cv2.putText(img, str(id), (int(pred['x']-pred['width']/2)-10, int(pred['y']-pred['height']/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 0, 255), 2, cv2.LINE_AA, False)
        img = cv2.putText(img, depth, (int(pred['x']+pred['width']/2)-100, int(pred['y']+pred['height']/2)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                        (0, 0, 255), 1, cv2.LINE_AA, False)
    
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float,dest='max_depth', default=20)
    
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
    
    margin_width=50
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        length = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        out_colormap = cv2.VideoWriter(args.colormap, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        
        frame=1
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            depth = depth_anything.infer_image(raw_frame, args.input_size)

            CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="LIBwQMSFtQcfjb2PHaKU"
            )   

            predictions = CLIENT.infer(raw_frame, model_id="test-2-g3mkp/15")

            depth_frame = get_object_depth(predictions['predictions'], depth, raw_frame)
            out.write(depth_frame)

            if args.savenumpy:
                output_path = os.path.join(args.savenumpy, os.path.splitext(os.path.basename(filename))[0] + '_numpy_matrix' + '_raw_depth_meter_frame' + str(frame) + '.npy')
                np.save(output_path, depth)

            if args.colormap:
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                out_colormap.write(depth)
            print(f'Frame: {frame}/{length} complete')
            frame+=1
        
        raw_video.release()