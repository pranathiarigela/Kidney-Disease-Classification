# src/components/model_trainer.py
import os
import sys
import json
import time
import shutil
import traceback
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

from utils.logger import log_info, log_error
from utils.exception import CustomException

class ModelTrainer:
    def __init__(self,
                 processed_dir: str = "data/processed",
                 models_dir: str = "models",
                 model_name: str = "kidney_model.pth",
                 batch_size: int = 32,
                 epochs: int = 20,
                 lr: float = 1e-4,
                 use_transfer_learning: bool = True,
                 target_size: Tuple[int,int] = (224, 224),
                 device: Optional[str] = None,
                 patience: int = 5,
                 verbose: bool = True):
        try:
            self.processed_dir = Path(processed_dir)
            self.models_dir = Path(models_dir)
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = self.models_dir / model_name

            self.batch_size = batch_size
            self.epochs = epochs
            self.lr = lr
            self.use_tl = use_transfer_learning
            self.target_size = target_size
            self.patience = patience
            self.verbose = verbose

            # device
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # numpy data files
            self.X_train_path = self.processed_dir / "X_train.npy"
            self.y_train_path = self.processed_dir / "y_train.npy"
            self.X_val_path = self.processed_dir / "X_val.npy"
            self.y_val_path = self.processed_dir / "y_val.npy"
            self.X_test_path = self.processed_dir / "X_test.npy"
            self.y_test_path = self.processed_dir / "y_test.npy"
            self.label_map_path = self.processed_dir / "label_map.json"

            if not self.label_map_path.exists():
                raise FileNotFoundError(f"label_map.json not found at {self.label_map_path}")

            log_info(f"ModelTrainer initialized. Device: {self.device}")
        except Exception as e:
            log_error(f"Error initializing ModelTrainer: {e}")
            raise CustomException("Failed to initialize ModelTrainer", sys)

    def _load_numpy_data(self, path_x: Path, path_y: Path):
        try:
            X = np.load(path_x)
            y = np.load(path_y)
            return X, y
        except Exception as e:
            log_error(f"Failed loading numpy arrays {path_x}, {path_y}: {e}")
            raise CustomException("Failed loading numpy arrays", sys)

    def _prepare_dataloaders(self):
        try:
            X_train, y_train = self._load_numpy_data(self.X_train_path, self.y_train_path)
            X_val, y_val = self._load_numpy_data(self.X_val_path, self.y_val_path)

            # If y is one-hot, convert to label indices
            if y_train.ndim == 2 and y_train.shape[1] > 1:
                y_train = np.argmax(y_train, axis=1)
            if y_val.ndim == 2 and y_val.shape[1] > 1:
                y_val = np.argmax(y_val, axis=1)

            # convert to torch tensors: (N,H,W,3) -> (N,3,H,W)
            X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            X_val_t = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
            y_val_t = torch.tensor(y_val, dtype=torch.long)

            train_ds = TensorDataset(X_train_t, y_train_t)
            val_ds = TensorDataset(X_val_t, y_val_t)

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

            log_info(f"Dataloaders prepared: train {len(train_ds)} samples, val {len(val_ds)} samples")
            return train_loader, val_loader
        except CustomException:
            raise
        except Exception as e:
            log_error(f"Error preparing dataloaders: {traceback.format_exc()}")
            raise CustomException("Error preparing dataloaders", sys)

    def _get_num_classes(self) -> int:
        try:
            with open(self.label_map_path, "r") as f:
                label_map = json.load(f)
            return len(label_map)
        except Exception as e:
            log_error(f"Failed reading label_map: {e}")
            raise CustomException("Failed reading label_map", sys)

    def _build_small_cnn(self, num_classes: int):
        # simple small CNN
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        return model

    def _build_transfer_model(self, num_classes: int):
        # mobilenet_v2 backbone
        try:
            backbone = models.mobilenet_v2(pretrained=True)
        except Exception:
            # older/newer torchvision may use weights=... fallback
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False
        # replace classifier
        in_features = backbone.classifier[1].in_features if hasattr(backbone, "classifier") else 1280
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        return backbone

    def _evaluate_loader(self, model, loader, criterion):
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss_sum += loss.item() * xb.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        avg_loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def run(self):
        """
        Train the model and save checkpoints and metrics.
        """
        try:
            num_classes = self._get_num_classes()
            train_loader, val_loader = self._prepare_dataloaders()

            # build model
            if self.use_tl:
                try:
                    model = self._build_transfer_model(num_classes)
                    log_info("Using MobileNetV2 transfer learning model (backbone frozen)")
                except Exception as e:
                    log_error(f"TL build failed, falling back. {e}")
                    model = self._build_small_cnn(num_classes)
            else:
                model = self._build_small_cnn(num_classes)
                log_info("Using small CNN model")

            model = model.to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

            best_val_loss = float("inf")
            best_epoch = -1
            epochs_no_improve = 0
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            for epoch in range(1, self.epochs + 1):
                model.train()
                running_loss = 0.0
                running_correct = 0
                running_total = 0
                t0 = time.time()
                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * xb.size(0)
                    preds = outputs.argmax(dim=1)
                    running_correct += (preds == yb).sum().item()
                    running_total += xb.size(0)

                train_loss = running_loss / running_total
                train_acc = running_correct / running_total

                val_loss, val_acc = self._evaluate_loader(model, val_loader, criterion)

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                elapsed = time.time() - t0
                log_info(f"Epoch {epoch}/{self.epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, time: {elapsed:.1f}s")
                if self.verbose:
                    print(f"Epoch {epoch}/{self.epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

                # checkpoint best
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    # save best model
                    torch.save(model.state_dict(), str(self.model_path))
                    log_info(f"Saved best model at epoch {epoch} to {self.model_path}")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    log_info(f"Early stopping triggered after {epoch} epochs (no improvement in {self.patience} epochs)")
                    break

            # load best model for final eval (if exists)
            if self.model_path.exists():
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))

            # final evaluation on val and test (if test exists)
            val_loss, val_acc = self._evaluate_loader(model, val_loader, criterion)
            test_loss, test_acc = None, None
            if self.X_test_path.exists() and self.y_test_path.exists():
                X_test, y_test = self._load_numpy_data(self.X_test_path, self.y_test_path)
                if y_test.ndim == 2 and y_test.shape[1] > 1:
                    y_test = np.argmax(y_test, axis=1)
                X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0,3,1,2)
                y_test_t = torch.tensor(y_test, dtype=torch.long)
                test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=self.batch_size, shuffle=False)
                test_loss, test_acc = self._evaluate_loader(model, test_loader, criterion)

            # save label_map to models folder
            try:
                shutil.copy(self.label_map_path, self.models_dir / "label_map.json")
            except Exception:
                pass

            metrics = {
                "best_epoch": best_epoch,
                "val_loss": float(val_loss) if val_loss is not None else None,
                "val_accuracy": float(val_acc) if val_acc is not None else None,
                "test_loss": float(test_loss) if test_loss is not None else None,
                "test_accuracy": float(test_acc) if test_acc is not None else None,
                "epochs_ran": epoch
            }
            # write metrics
            with open(self.models_dir / "metrics.txt", "w") as f:
                json.dump(metrics, f, indent=2)

            log_info("Training completed")
            return {"model_path": str(self.model_path), "metrics": metrics, "history": history}
        except CustomException:
            raise
        except Exception as e:
            log_error(f"Training failed: {traceback.format_exc()}")
            raise CustomException("Training failed", sys)


if __name__ == "__main__":
    trainer = ModelTrainer(epochs=10, batch_size=16, use_transfer_learning=True)
    out = trainer.run()
    print("Training result:", out)
