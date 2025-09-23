import cv2
import glob
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image

from unidepth.models import UniDepthV2
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation


class UniDepthV2Inference:
    """
    Single-model monocular depth inference using UniDepthV2, integrated with YOLO detection
    and DepthApproximation utilities (matches style of your DepthAnythingV2Inference).
    """

    def __init__(
        self,
        input_path,
        model_name="vitl14",
        outdir="results/unidepth_v2",
        max_depth=80,
        savenumpy=False,              # either False or a path where .npy files will be written
        colormap="",                  # either "" or a directory path to save colormap PNGs
        eval=False,
        depth_path="",
        stream=False,
        device=None,
    ):
        """
        Args:
            input_path: single image / video file or a directory with images
            model_name: one of 'vits14', 'vitb14', 'vitl14'
            outdir: output directory for annotated frames/videos/results
            max_depth: used by DepthApproximation for scaling/annotations
            savenumpy: False or directory path for raw .npy depth arrays
            colormap: Falsey or directory path to save colored depth visualizations
            eval: whether to run evaluation routines when available
            depth_path: path to ground-truth depths (used by evaluation routines)
            stream: reserved for future stream handling (kept for API parity)
            device: 'cuda', 'mps', or 'cpu' (auto-selected if None)
        """
        self.input_path = input_path
        self.outdir = outdir
        self.model_name = model_name
        self.max_depth = max_depth
        self.savenumpy = savenumpy
        self.colormap = colormap
        self.evaluate = eval
        self.depth_path = depth_path
        self.stream = stream

        # device selection
        if device is not None:
            self.DEVICE = device
        else:
            self.DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        # Validate model_name
        allowed = {"vits14", "vitb14", "vitl14"}
        if self.model_name not in allowed:
            raise ValueError(f"model_name must be one of {allowed}")

        # create outdir
        os.makedirs(self.outdir, exist_ok=True)
        if self.savenumpy:
            os.makedirs(self.savenumpy, exist_ok=True)
        if self.colormap:
            os.makedirs(self.colormap, exist_ok=True)

        # Load UniDepthV2
        try:
            model_id = f"lpiccinelli/unidepth-v2-{self.model_name}"
            print(f"Loading UniDepthV2 model '{model_id}' on device '{self.DEVICE}' ...")
            self.unidepth = UniDepthV2.from_pretrained(model_id).to(self.DEVICE).eval()
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading UniDepthV2 model: {e}")
            raise

        # Initialize object detection and depth approximation utilities
        # (paths follow your original pattern; swap if you use different weights)
        self.model = Yolo2DObjectDetection("models/detection_models/bounding_box/yolov8n.pt")
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)

        # collect file list
        self.process_files()

    def process_files(self):
        """Gather files from input_path. If file, single entry; else scan directory for images."""
        if os.path.isfile(self.input_path):
            self.filenames = [self.input_path]
        else:
            all_files = glob.glob(os.path.join(self.input_path, "**/*"), recursive=True)
            image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            self.filenames = [f for f in all_files if f.lower().endswith(image_extensions)]
        print(f"Found {len(self.filenames)} image files to process")

    def _preprocess_bgr_to_tensor(self, bgr_image):
        """Convert OpenCV BGR image to UniDepth expected tensor on device."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0  # normalize to [0,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.DEVICE)  # 1xCxHxW
        return tensor

    def infer_depth(self, image):
        """
        Infer depth from a single OpenCV BGR image using UniDepthV2.
        Returns a HxW numpy array of depth (meters-like scale depending on model).
        """
        tensor = self._preprocess_bgr_to_tensor(image)
        with torch.inference_mode():
            preds = self.unidepth.infer(tensor)
            depth = preds.get("depth", None)
            if depth is None:
                raise RuntimeError("UniDepth inference did not return 'depth' key")
            # ensure numpy H x W
            if torch.is_tensor(depth):
                depth_np = depth.squeeze().cpu().numpy()
            else:
                depth_np = np.asarray(depth)
        return depth_np

    def _safe_colormap_and_save(self, depth_np, filename_base):
        """Normalize depth safely, apply a matplotlib colormap, and save as PNG (BGR)"""
        # Avoid division by zero and invalid values
        if np.nanmin(depth_np) == np.nanmax(depth_np):
            norm = np.zeros_like(depth_np, dtype=np.float32)
        else:
            dmin = np.nanmin(depth_np)
            dmax = np.nanmax(depth_np)
            norm = (depth_np - dmin) / (dmax - dmin + 1e-8)

        vis = (norm * 255.0).astype(np.uint8)

        # Use OpenCV colormap (MAGMA-like). You can swap to other colormaps if desired.
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)

        out_path = os.path.join(self.colormap, f"{filename_base}_unidepth_colormap.png")
        cv2.imwrite(out_path, colored)

    def process_images(self):
        """Process all images in self.filenames, run detection, approximations and save outputs."""
        for k, filename in enumerate(self.filenames):
            print(f"Progress {k+1}/{len(self.filenames)}: {filename}")

            raw_image = cv2.imread(filename)
            if raw_image is None:
                print(f"  -> Could not read image: {filename}, skipping.")
                continue

            # UniDepth inference
            try:
                depth = self.infer_depth(raw_image)
            except Exception as e:
                print(f"  -> Depth inference failed for {filename}: {e}")
                continue

            # Object detection
            predictions = self.model.predict(raw_image)

            # Combine depth with detections
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_image, predictions)

            # Optional evaluation
            if self.evaluate:
                depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)

            # Save annotated image
            output_path = os.path.join(self.outdir, f"frame_{k}.png")
            cv2.imwrite(output_path, depth_frame)

            # Save numpy array (if savenumpy is a path string)
            if self.savenumpy:
                npy_output_path = os.path.join(
                    self.savenumpy,
                    os.path.splitext(os.path.basename(filename))[0] + "_unidepth_raw_depth_meter.npy",
                )
                os.makedirs(os.path.dirname(npy_output_path), exist_ok=True)
                np.save(npy_output_path, depth)

            # Save colormap (if requested)
            if self.colormap:
                try:
                    self._safe_colormap_and_save(depth, os.path.splitext(os.path.basename(filename))[0])
                except Exception as e:
                    print(f"  -> Failed saving colormap for {filename}: {e}")

        print(f"Output saved to {self.outdir}")

    def process_video(self, fps=None, process_every_n_frames=1):
        """
        Process a video file (or files). The input_path should be a video file or self.filenames[0] used.
        process_every_n_frames: process only every Nth frame (1=every frame)
        """
        # If input was a directory, user must pass a video file; otherwise process the listed files
        for k, filename in enumerate(self.filenames):
            if not os.path.isfile(filename):
                print(f"  -> File not found (skipping): {filename}")
                continue

            cap = cv2.VideoCapture(filename)
            if not cap.isOpened():
                print(f"  -> Could not open video: {filename}")
                continue

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            out_fps = (input_fps / max(1, process_every_n_frames)) if fps is None else fps

            output_path = os.path.join(self.outdir, os.path.splitext(os.path.basename(filename))[0] + "_unidepth.mp4")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (frame_width, frame_height))

            frame_idx = 0
            saved_frames = 0
            print(f"Processing video {k+1}/{len(self.filenames)}: {filename}  (total frames: {length})")

            while True:
                ret, raw_frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if (frame_idx - 1) % process_every_n_frames != 0:
                    continue

                # depth inference
                try:
                    depth = self.infer_depth(raw_frame)
                except Exception as e:
                    print(f"  -> Depth inference failed at frame {frame_idx}: {e}")
                    continue

                predictions = self.model.predict(raw_frame)
                predictions = self.depth_approximator.depth(predictions, depth)
                depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)

                if self.evaluate:
                    depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)

                out.write(depth_frame)
                saved_frames += 1

                # Optionally save per-frame numpy and colormap
                if self.savenumpy:
                    npy_name = os.path.join(
                        self.savenumpy,
                        f"{os.path.splitext(os.path.basename(filename))[0]}_frame_{frame_idx:06d}_depth.npy",
                    )
                    os.makedirs(os.path.dirname(npy_name), exist_ok=True)
                    np.save(npy_name, depth)

                if self.colormap:
                    try:
                        base = f"{os.path.splitext(os.path.basename(filename))[0]}_frame_{frame_idx:06d}"
                        self._safe_colormap_and_save(depth, base)
                    except Exception as e:
                        print(f"  -> Failed saving colormap for frame {frame_idx}: {e}")

                print(f"Frame: {frame_idx}/{length} processed")

            cap.release()
            out.release()
            print(f"Video output saved to {output_path} ({saved_frames} frames written)")

        print(f"All video outputs saved to {self.outdir}")

    def eval_stereo(self, timestamps_csv_path=None, max_frames=20):
        """
        Evaluate predicted depths against stereo/ground truth if available.
        This mirrors the pattern in your DepthAnythingV2Inference.eval_stereo:
        - expects frames named 'frame_{index}.jpg' in input_path OR uses self.filenames
        - writes a CSV with evaluation results
        """
        data = []
        df_ts = None
        if timestamps_csv_path is None:
            # try default location next to input_path
            ts_path = os.path.join(os.path.dirname(self.input_path), "timestamps.csv")
        else:
            ts_path = timestamps_csv_path

        if os.path.exists(ts_path):
            try:
                df_ts = pd.read_csv(ts_path)
            except Exception as e:
                print(f"  -> Failed to read timestamps CSV at {ts_path}: {e}")

        # iterate through found filenames (or construct frame names like your original code)
        for k, filename in enumerate(self.filenames):
            # if filenames are not in the frame_x pattern and you expect that pattern, adjust accordingly
            print(f"Progress {k+1}/{len(self.filenames)}: {filename}")

            raw_frame = cv2.imread(filename)
            if raw_frame is None:
                print(f"  -> Could not read {filename}, skipping.")
                continue

            depth = self.infer_depth(raw_frame)
            predictions = self.model.predict(raw_frame)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)

            # Evaluate (should return annotated frame and maybe updated predictions)
            try:
                depth_frame, predictions = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
            except Exception as e:
                # If evaluate is not implemented to return both, try catching that case
                try:
                    depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
                except Exception:
                    print(f"  -> Evaluation function failed or is not available: {e}")

            # Save annotated frame
            output_path = os.path.join(self.outdir, os.path.splitext(os.path.basename(filename))[0] + "_eval.png")
            cv2.imwrite(output_path, depth_frame)

            # Collect evaluation rows per detection
            for pred in predictions:
                if df_ts is not None:
                    # attempt to lookup timestamp by a frame_number column
                    frame_number = pred.get("frame_number", None)
                    if frame_number is None and "frame" in filename:
                        # naive attempt to extract integer from filename
                        try:
                            frame_number = int("".join(filter(str.isdigit, os.path.splitext(os.path.basename(filename))[0])))
                        except Exception:
                            frame_number = None

                    timestamp = "N/A"
                    if frame_number is not None:
                        rows = df_ts[df_ts.get("frame_number", -1) == frame_number]
                        if not rows.empty:
                            timestamp = rows["utc_timestamp"].iloc[0]

                else:
                    timestamp = "N/A"

                res = [
                    os.path.splitext(os.path.basename(filename))[0],
                    timestamp,
                    pred.get("x1", "N/A"),
                    pred.get("y1", "N/A"),
                    pred.get("x2", "N/A"),
                    pred.get("y2", "N/A"),
                    pred.get("class", "N/A"),
                    pred.get("estimated_depth", "N/A"),
                    pred.get("actual_depth", "N/A"),
                ]
                data.append(res)

            if (k + 1) >= max_frames:
                break

        result_df = pd.DataFrame(
            data,
            columns=["Frame", "UTC_time", "x1", "y1", "x2", "y2", "CLASS", "predicted_depth", "actual_depth"],
        )

        csv_path = os.path.join(self.outdir, "unidepth_v2_results.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"Evaluation results saved to {csv_path}")


# # Example usage (uncomment to run as script)
# if __name__ == "__main__":
#     # Minimal demo — edit paths as required
#     inference = UniDepthV2Inference(
#         input_path="path/to/images_or_video_or_directory",
#         model_name="vitl14",
#         outdir="results/unidepth_v2",
#         max_depth=80,
#         savenumpy=False,      # or "results/numpy_outputs"
#         colormap=False,       # or "results/colormaps"
#         eval=False,
#     )

#     # If single file that is a video, call process_video(); otherwise process_images()
#     input_path = inference.input_path
#     video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
#     if os.path.isfile(input_path) and input_path.lower().endswith(video_exts):
#         inference.process_video(process_every_n_frames=1)
#     else:
#         inference.process_images()
