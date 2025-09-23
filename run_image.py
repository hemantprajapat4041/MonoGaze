import argparse
import os
import sys


# from depth_inference.depth_anything_v2 import DepthAnythingV2Inference
# from depth_inference.monodepth2 import MonoDepth2Inference
# from depth_inference.midas_inference import MiDaSInference
# from depth_inference.unidepth import UniDepthInference
from depth_inference.zeodepth_inference import ZeoDepthInference
# from depth_inference.marigold import MarigoldInference
# from depth_inference.depthFm import DepthFMInference
# from depth_inference.hrdepth_inference import HRDepthInference
# from depth_inference.metric3d_inference import Metric3DInference
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Estimation Script')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input image or directory of images')

    args = parser.parse_args()
    print("Script started")

    # depth_anything = DepthAnythingV2Inference(input_path=args.input_path)
    # monodepth = MonoDepth2Inference(input_path=args.input_path)

    # midas = MiDaSInference(input_path=args.input_path)
    # unidepth = UniDepthInference(input_path=args.input_path)
    zeodepth = ZeoDepthInference(input_path=args.input_path)
    # marigold = MarigoldInference(input_path=args.input_path)
    # depthfm = DepthFMInference(input_path=args.input_path)
    # hrdepth= HRDepthInference(input_path=args.input_path)
    # metric3d = Metric3DInference(input_path=args.input_path)
    print("Models initialized")
    # depth_anything.process_images()
    # monodepth.process_image()
    # midas.process_images()
    # unidepth.process_images()
    zeodepth.process_images()
    # marigold.process_images()
    # depthfm.process_images()
    # hrdepth.process_images()
    # metric3d.process_images()
    print("Script finished")
