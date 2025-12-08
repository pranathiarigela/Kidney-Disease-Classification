# src/pipeline/prediction_pipeline.py
import os
import sys
import json
from pathlib import Path
from typing import Union, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from utils.logger import log_info, log_error
from utils.exception import CustomException
from src.components.model_trainer import ModelTrainer

class PredictionPipeline:
    def __init__(self,
                 models_dir: str = "models",
                 processed_dir: str = "data/processed",
                 model_name: str = "kidney_model.pth",
                 target_size: tuple = (224, 224),
                 device: str = None):
        try:
            self.models_dir = Path(models_dir)
            self.processed_dir = Path(processed_dir)
            self.model_path = self.models_dir / model_name
            self.label_map_path = self.models_dir / "label_map.json" if (self.models_dir / "label_map.json").exists() else (self.processed_dir / "label_map.json")

            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            if not self.label_map_path.exists():
                raise FileNotFoundError(f"label_map.json not found at {self.label_map_path}")

            with open(self.label_map_path, "r") as f:
                self.label_map = json.load(f)  # e.g. {"normal":0, "cyst":1,...}
            # invert map: idx -> class
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
            self.num_classes = len(self.label_map)

            self.target_size = target_size

            # build model structure using ModelTrainer helper to ensure match
            trainer = ModelTrainer(processed_dir=str(self.processed_dir),
                                   models_dir=str(self.models_dir),
                                   model_name=model_name)
            # try transfer model build first (same default as trainer)
            try:
                self.model = trainer._build_transfer_model(self.num_classes)
            except Exception:
                self.model = trainer._build_small_cnn(self.num_classes)

            # load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # checkpoint might be a full checkpoint dict or state_dict directly
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = checkpoint
            try:
                self.model.load_state_dict(state_dict)
            except Exception as e:
                # try loading with strict=False (in case of minor mismatch)
                self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()
            log_info(f"Loaded model for prediction from {self.model_path} on device {self.device}")

        except Exception as e:
            log_error(f"Failed initializing PredictionPipeline: {e}")
            raise CustomException("Failed to initialize PredictionPipeline", sys)

    def _preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Accepts a file path or PIL.Image, returns a tensor (1,3,H,W) on self.device.
        """
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(str(image)).convert("RGB")
            elif isinstance(image, Image.Image):
                img = image.convert("RGB")
            else:
                raise ValueError("Input must be a file path or PIL.Image")

            img = img.resize(self.target_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize same as preprocessing
            tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1,3,H,W)
            return tensor
        except Exception as e:
            log_error(f"Preprocess failed: {e}")
            raise CustomException("Failed preprocessing input image", sys)

    def predict(self, image: Union[str, Image.Image]) -> Dict[str, Union[str, float]]:
        """
        Returns:
            {"class": "<label_name>", "confidence": 0.934}
        """
        try:
            x = self._preprocess(image)
            with torch.no_grad():
                outputs = self.model(x)
                probs = F.softmax(outputs, dim=1).cpu().numpy().squeeze()
                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])
                label_name = self.inv_label_map.get(top_idx, str(top_idx))
            return {"class": label_name, "confidence": top_conf}
        except Exception as e:
            log_error(f"Prediction failed: {e}")
            raise CustomException("Prediction failed", sys)


# convenience function
def predict_from_path(image_path: str, device: str = None) -> Dict:
    pp = PredictionPipeline(device=device)
    return pp.predict(image_path)


if __name__ == "__main__":
    # quick manual test
    tester = PredictionPipeline()
    sample = "data/processed/test/"  # replace with a specific test image path or open one from processed folders
    # find first image in processed test folder
    import glob
    files = glob.glob(str(Path("data/processed/test") / "*" / "*"))
    if files:
        out = tester.predict(files[0])
        print("Prediction:", out)
    else:
        print("No test images found. Provide a path to predict.")
