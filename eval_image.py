import argparse
from depth_inference.depth_anything_v2 import DepthAnythingV2Inference
from depth_inference.monodepth2 import MonoDepth2Inference
from depth_inference.unidepth import UniDepthInference
from depth_inference.zeodepth_inference import ZeoDepthInference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Estimation Script')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input image or directory of images')
    parser.add_argument('--eval',type=bool, default=False, help='Evaluate the model')
    parser.add_argument('--depth-path', type=str, help='Path to depth image or directory of depth images')

    args = parser.parse_args()

    # depth_anything = DepthAnythingV2Inference(input_path=args.input_path, depth_path=args.depth_path, eval=args.eval)
    # monodepth = MonoDepth2Inference(input_path=args.input_path)
    # unidepth = UniDepthInference(input_path=args.input_path, depth_path=args.depth_path, eval=args.eval)
    zeodepth = ZeoDepthInference(input_path=args.input_path, depth_path=args.depth_path, eval=args.eval)

    # depth_anything.eval_stereo()
    # monodepth.process_image()
    # unidepth.eval_stereo()
    zeodepth.eval_stereo()


    