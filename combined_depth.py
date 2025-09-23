import os
import glob
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from depth_inference.zeodepth_inference import ZeoDepthInference
from depth_inference.depth_anything_v2 import DepthAnythingV2Inference
from depth_inference.hrdepth_inference import HRDepthInference
from depth_inference.unidepth import UniDepthInference
from depth_inference.depth_model_comparison import iou, bbox_from_pred, CombinedDepthEvaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True, help='Folder with images')
    # parser.add_argument('--out', default='results/combined_depths.xlsx', help='Output excel file')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--iou', type=float, default=0.5)
    args = parser.parse_args()

    evaluator = CombinedDepthEvaluator(args.images, args.out, max_images=args.max_images, iou_threshold=args.iou)
    evaluator.run()