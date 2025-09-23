import cv2
import glob
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

# Import MiDaS model
from models.depth_models.MiDaS.midas.dpt_depth import DPTDepthModel
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation


class MiDaSInference:
    def __init__(self, input_path, model_path='models/depth_models/MiDaS/weights/dpt_beit_large_512.pt',
                 input_size=512, outdir='results/midas', max_depth=80, savenumpy=False, colormap='',
                 eval=False, depth_path='', stream=False):
        print("🔄 Initializing MiDaSInference...")
        self.input_path = input_path
        self.input_size = input_size
        self.outdir = outdir
        self.model_path = model_path.replace("\\", "/")  # safer on Windows
        self.max_depth = max_depth
        self.evalualte = eval
        self.depth_path = depth_path
        self.stream = stream
        self.savenumpy = savenumpy
        self.colormap = colormap

        # Device setup
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Using device: {self.DEVICE}")

        # Model loading
        print(f"📂 Loading MiDaS model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model weights not found at {self.model_path}")

        self.model = DPTDepthModel(path=self.model_path, backbone="beitl16_512")
        self.model = self.model.to(self.DEVICE).eval()
        print("✅ MiDaS model loaded successfully")

        # Object detector
        print("📂 Loading YOLO detector...")
        self.detector = Yolo2DObjectDetection('models/detection_models/bounding_box/yolov8n.pt')
        print("✅ YOLO detector initialized")

        # Depth approximator
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)
        print("✅ DepthApproximation ready")

        # File processing
        self.process_files()
        print(f"📂 Found {len(self.filenames)} file(s) to process")

        # Image preprocessing transform
        self.transform = Compose([
            Resize((self.input_size, self.input_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("✅ Transform pipeline ready")

    def process_files(self):
        if os.path.isfile(self.input_path):
            self.filenames = [self.input_path]
        else:
            self.filenames = glob.glob(os.path.join(self.input_path, '**/*'), recursive=True)
        os.makedirs(self.outdir, exist_ok=True)

    def infer_image(self, img):
        print("🔄 Running depth inference on image...")
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = depth.cpu().numpy().squeeze()
        print("✅ Depth inference complete")
        return depth

    def process_images(self):
        for k, filename in enumerate(self.filenames):
            print(f"\n➡️ Progress {k+1}/{len(self.filenames)}: {filename}")
            raw_image = cv2.imread(filename)
            if raw_image is None:
                print(f"❌ Failed to read image: {filename}")
                continue

            # Depth inference
            depth = self.infer_image(raw_image)
            output_path = os.path.join(self.outdir, f'frame_{k}.png')

            # Detection + depth
            print("🔄 Running object detection...")
            predictions = self.detector.predict(raw_image)
            print(f"✅ Detected {len(predictions)} objects")

            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_image, predictions)

            if self.evalualte:
                depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)

            cv2.imwrite(output_path, depth_frame)
            print(f"💾 Saved annotated frame to {output_path}")

            # Save numpy depth
            if self.savenumpy:
                np.save(os.path.join(self.savenumpy, f'frame_{k}_depth.npy'), depth)
                print(f"💾 Saved raw depth to {self.savenumpy}")

            # Save colormap
            if self.colormap:
                depth_vis = (255 * (depth - depth.min()) / (depth.max() - depth.min())).astype("uint8")
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                cv2.imwrite(os.path.join(self.colormap, f'frame_{k}_colormap.png'), depth_vis)
                print(f"💾 Saved colormap visualization to {self.colormap}")

        print(f"\n✅ All outputs saved to {self.outdir}")
