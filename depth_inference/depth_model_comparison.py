"""
combine_depths.py

Reads images from a folder, runs four depth models (ZoeDepth, DepthAnythingV2, HRDepth, UniDepth),
matches detections across model outputs, computes per-object predicted depths from each model,
computes mean prediction and writes a single Excel file with coordinates, per-model depths,
mean depth and actual depth (if available).

Usage:
    python combine_depths.py --images /path/to/images --out results/combined.xlsx

Notes:
 - This script expects the four inference modules to be importable in the same project (zeodepth_inference.py,
   depth_anything_v2.py, hrdepth_inference.py, unidepth.py).
 - For `actual_depth` to be filled, the detection->depth evaluation in DepthApproximation must attach
   'actual_depth' to predictions (this usually happens when you enable evaluation and provide depth/GT path).
 - Running these models can be GPU / memory heavy. You can reduce workload by passing --max_images N.
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import your four inference classes (adjust module paths if needed)
from zeodepth_inference import ZeoDepthInference
from depth_anything_v2 import DepthAnythingV2Inference
from hrdepth_inference import HRDepthInference
from unidepth import UniDepthInference


def iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    denom = float(boxAArea + boxBArea - interArea)
    if denom == 0:
        return 0.0
    return interArea / denom


def bbox_from_pred(pred):
    # adapt to how Yolo2DObjectDetection returns preds in your repo
    # expecting pred to have keys 'x1','y1','x2','y2' or be a list/tuple
    if isinstance(pred, dict):
        return [pred.get('x1'), pred.get('y1'), pred.get('x2'), pred.get('y2')]
    elif hasattr(pred, '__len__') and len(pred) >= 4:
        return [pred[0], pred[1], pred[2], pred[3]]
    else:
        return None


class CombinedDepthEvaluator:
    def __init__(self, input_path, outdir="results/combined", max_images=None, iou_threshold=0.5, device=None, model_args=None):
        self.images_dir = input_path
        self.outdir=outdir
        self.max_images = max_images
        self.iou_threshold = iou_threshold
        self.model_args = model_args or {}

        # Instantiate models
        print("Loading models (this may take a while)...")
        # Pass any required args (paths, encoders) via model_args dict
        self.zeo = ZeoDepthInference(input_path, **self.model_args.get('zeo', {}))
        self.dav2 = DepthAnythingV2Inference(input_path, **self.model_args.get('dav2', {}))
        self.hr = HRDepthInference(input_path, **self.model_args.get('hr', {}))
        self.uni = UniDepthInference(input_path, **self.model_args.get('uni', {}))

    def run(self):
        # list images
        images = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        images.sort()
        if self.max_images:
            images = images[:self.max_images]

        rows = []

        for img_path in tqdm(images, desc='Images'):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # get depth maps
            depth_zeo = self.zeo.infer_depth(img)
            depth_dav2 = self.dav2.depth_anything.infer_image(img, self.dav2.input_size) if hasattr(self.dav2, 'depth_anything') else self.dav2.depth_anything.infer_image(img, self.dav2.input_size)
            depth_hr = self.hr.infer_depth(img)
            depth_uni = self.uni.infer_depth(img)

            # get detections (YOLO predictions) from models (they use same detection wrapper so should be comparable)
            preds_zeo = self.zeo.model.predict(img)
            preds_dav2 = self.dav2.model.predict(img)
            preds_hr = self.hr.model.predict(img)
            preds_uni = self.uni.model.predict(img)

            # Convert each predictions list to a uniform list of dicts with bbox,class
            def norm_preds(pred_list, depth_map):
                out = []
                for p in pred_list:
                    bbox = bbox_from_pred(p)
                    if bbox is None:
                        continue
                    # estimated depth extraction: DepthApproximation.depth() normally sets 'estimated_depth' if called.
                    # but here we compute estimated depth as median of depth_map inside bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(depth_map.shape[1]-1, x2), min(depth_map.shape[0]-1, y2)
                    if x2c <= x1c or y2c <= y1c:
                        est_depth = np.nan
                    else:
                        crop = depth_map[y1c:y2c+1, x1c:x2c+1]
                        if crop.size == 0:
                            est_depth = np.nan
                        else:
                            est_depth = float(np.nanmedian(crop))

                    out.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'class': p.get('class') if isinstance(p, dict) else None,
                                'estimated_depth': est_depth,
                                'raw': p})
                return out

            n_zeo = norm_preds(preds_zeo, depth_zeo)
            n_dav2 = norm_preds(preds_dav2, depth_dav2)
            n_hr = norm_preds(preds_hr, depth_hr)
            n_uni = norm_preds(preds_uni, depth_uni)

            # Use Zoe detections as reference; match others by IoU
            for ref in n_zeo:
                row = {
                    'image': os.path.basename(img_path),
                    'x1': ref['x1'], 'y1': ref['y1'], 'x2': ref['x2'], 'y2': ref['y2'],
                    'class': ref.get('class')
                }
                depths = {}
                depths['zeo'] = ref['estimated_depth']

                # helper to find match
                def find_match(ref_bbox, candidates):
                    best = None
                    best_iou = 0.0
                    for c in candidates:
                        bb = [c['x1'], c['y1'], c['x2'], c['y2']]
                        val = iou(ref_bbox, bb)
                        if val > best_iou:
                            best_iou = val
                            best = c
                    if best_iou >= self.iou_threshold:
                        return best
                    return None

                ref_bbox = [ref['x1'], ref['y1'], ref['x2'], ref['y2']]
                m = find_match(ref_bbox, n_dav2)
                depths['dav2'] = m['estimated_depth'] if m is not None else np.nan
                m = find_match(ref_bbox, n_hr)
                depths['hr'] = m['estimated_depth'] if m is not None else np.nan
                m = find_match(ref_bbox, n_uni)
                depths['uni'] = m['estimated_depth'] if m is not None else np.nan

                # mean across available model predictions
                arr = np.array([depths.get(k) for k in ['zeo', 'dav2', 'hr', 'uni']], dtype=float)
                mean_pred = float(np.nanmean(arr)) if np.count_nonzero(~np.isnan(arr)) > 0 else np.nan

                row.update({
                    'depth_zeo': depths['zeo'],
                    'depth_dav2': depths['dav2'],
                    'depth_hr': depths['hr'],
                    'depth_uni': depths['uni'],
                    'mean_predicted_depth': mean_pred,
                    # actual depth: try to read from any raw pred dict if available
                    'actual_depth': ref.get('raw', {}).get('actual_depth') if isinstance(ref.get('raw'), dict) else np.nan
                })

                rows.append(row)

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(self.out_excel) or '.', exist_ok=True)
        df.to_excel(self.out_excel, index=False)
        print(f"Saved combined results to {self.out_excel}")



