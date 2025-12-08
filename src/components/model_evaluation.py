import os
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.logger import log_info, log_error
from utils.exception import CustomException


class ModelEvaluation:
    def __init__(self,
                 processed_dir="data/processed",
                 models_dir="models",
                 model_name="kidney_model.pth",
                 batch_size=32,
                 device=None):
        try:
            self.processed_dir = Path(processed_dir)
            self.models_dir = Path(models_dir)
            self.model_path = self.models_dir / model_name
            self.batch_size = batch_size

            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.label_map_path = self.models_dir / "label_map.json"

            if not self.model_path.exists():
                raise FileNotFoundError("Trained model not found.")

            if not self.label_map_path.exists():
                raise FileNotFoundError("label_map.json missing.")

            log_info("ModelEvaluation initialized.")
        except Exception as e:
            log_error(f"Error initializing ModelEvaluation: {e}")
            raise CustomException("Failed to initialize ModelEvaluation", sys)

    def _load_test_data(self):
        try:
            X_test = np.load(self.processed_dir / "X_test.npy")
            y_test = np.load(self.processed_dir / "y_test.npy")

            # Convert one-hot to index
            if y_test.ndim == 2 and y_test.shape[1] > 1:
                y_test = np.argmax(y_test, axis=1)

            X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
            y_test_t = torch.tensor(y_test, dtype=torch.long)

            test_ds = TensorDataset(X_test_t, y_test_t)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

            return test_loader, y_test
        except Exception as e:
            log_error(f"Failed loading test data: {e}")
            raise CustomException("Failed loading test data", sys)

    def _load_label_map(self):
        with open(self.label_map_path, "r") as f:
            label_map = json.load(f)
        inv_map = {v: k for k, v in label_map.items()}
        return label_map, inv_map

    def _load_model(self, num_classes):
        from src.components.model_trainer import ModelTrainer

        trainer = ModelTrainer()
        model = trainer._build_transfer_model(num_classes)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def evaluate(self):
        try:
            test_loader, y_true = self._load_test_data()
            label_map, inv_map = self._load_label_map()
            num_classes = len(label_map)

            model = self._load_model(num_classes)

            y_pred = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(self.device)
                    outputs = model(xb)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    y_pred.extend(preds)

            # Convert to numpy arrays
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

            # Metrics
            cls_report = classification_report(
                y_true, y_pred, target_names=[inv_map[i] for i in range(num_classes)],
                output_dict=True
            )

            conf_mat = confusion_matrix(y_true, y_pred)

            # Save detailed metrics
            metrics_path = self.models_dir / "metrics_detailed.json"
            with open(metrics_path, "w") as f:
                json.dump(cls_report, f, indent=2)

            log_info(f"Saved detailed metrics: {metrics_path}")

            # Save confusion matrix image
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_mat, cmap="Blues")
            plt.title("Confusion Matrix")
            plt.colorbar()

            classes = [inv_map[i] for i in range(num_classes)]
            plt.xticks(range(num_classes), classes, rotation=45)
            plt.yticks(range(num_classes), classes)

            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, conf_mat[i, j], ha="center", va="center", color="black")

            cm_path = self.models_dir / "confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()

            log_info(f"Saved confusion matrix: {cm_path}")

            return {
                "metrics_file": str(metrics_path),
                "confusion_matrix": str(cm_path)
            }

        except Exception as e:
            log_error(f"Model evaluation failed: {e}")
            raise CustomException("Model evaluation failed", sys)
