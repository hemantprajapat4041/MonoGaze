import argparse

from depth_inference.depth_anything_v2 import DepthAnythingV2Inference
from depth_inference.monodepth2 import MonoDepth2Inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Estimation Script')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input image or directory of images')

    args = parser.parse_args()

    depth_anything = DepthAnythingV2Inference(input_path=args.input_path)
    # monodepth = MonoDepth2Inference(input_path=args.input_path)

    depth_anything.process_images()
    # monodepth.process_video()

    