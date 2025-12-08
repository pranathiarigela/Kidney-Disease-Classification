import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from utils.logger import log_info, log_error
from utils.exception import CustomException

class DataPreprocessing:
    def __init__(self,
                 processed_dir: str = "data/processed",
                 target_size: Tuple[int,int] = (224, 224),
                 normalize: bool = True,
                 verbose: bool = True):
        try:
            self.processed_dir = Path(processed_dir)
            self.target_size = target_size
            self.normalize = normalize
            self.verbose = verbose

            # Required files
            self.train_csv = self.processed_dir / "train.csv"
            self.val_csv = self.processed_dir / "val.csv"
            self.test_csv = self.processed_dir / "test.csv"
            self.label_map_path = self.processed_dir / "label_map.json"

            if not self.label_map_path.exists():
                raise FileNotFoundError(f"label_map.json not found at {self.label_map_path}")

            log_info(f"Initialized DataPreprocessing with target_size={self.target_size}")
        except Exception as e:
            log_error(f"Failed initializing DataPreprocessing: {e}")
            raise CustomException("Failed to initialize DataPreprocessing", sys)

    def _load_label_map(self):
        try:
            with open(self.label_map_path, "r") as f:
                label_map = json.load(f)
            # create inverse map: idx -> class
            inv_map = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)
            return label_map, inv_map, num_classes
        except Exception as e:
            log_error(f"Error loading label map: {e}")
            raise CustomException("Error loading label map", sys)

    def _read_csv(self, csv_path: Path) -> List[Tuple[str, int]]:
        try:
            records = []
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fp = row.get("filepath") or row.get("path") or row.get("file")
                    label = row.get("label")
                    if fp is None or label is None:
                        continue
                    records.append((fp, int(label)))
            log_info(f"Loaded {len(records)} records from {csv_path}")
            return records
        except Exception as e:
            log_error(f"Error reading CSV {csv_path}: {e}")
            raise CustomException(f"Error reading CSV {csv_path}", sys)

    def _process_records(self, records: List[Tuple[str,int]], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images, resize, normalize, and convert labels to one-hot.
        Returns X (N,H,W,3) and y (N,num_classes)
        """
        images = []
        labels = []
        for idx, (fp, lbl) in enumerate(records):
            try:
                img = Image.open(fp).convert("RGB")
                img = img.resize(self.target_size, Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32)
                if self.normalize:
                    arr = arr / 255.0
                images.append(arr)
                labels.append(lbl)
            except Exception as e:
                # log and skip corrupted image
                log_error(f"Error processing image {fp}: {e}")
                continue
            if self.verbose and (idx + 1) % 500 == 0:
                log_info(f"Processed {idx + 1} images")

        if not images:
            raise CustomException("No images were processed; check your CSV paths", sys)

        X = np.stack(images, axis=0)
        y = np.array(labels, dtype=np.int32)
        # one-hot
        y_onehot = np.eye(num_classes)[y]
        return X, y_onehot

    def run(self):
        """
        Main runner that processes train, val, test and saves numpy arrays.
        Files saved:
          - data/processed/X_train.npy, y_train.npy
          - data/processed/X_val.npy, y_val.npy
          - data/processed/X_test.npy, y_test.npy
        """
        try:
            label_map, inv_map, num_classes = self._load_label_map()

            results = {}
            for split_name, csv_path in [("train", self.train_csv),
                                         ("val", self.val_csv),
                                         ("test", self.test_csv)]:
                if not csv_path.exists():
                    log_info(f"No {split_name}.csv found at {csv_path}, skipping.")
                    results[split_name] = {"X_shape": None, "y_shape": None, "saved": False}
                    continue

                records = self._read_csv(csv_path)
                X, y = self._process_records(records, num_classes)

                x_out = self.processed_dir / f"X_{split_name}.npy"
                y_out = self.processed_dir / f"y_{split_name}.npy"

                np.save(x_out, X)
                np.save(y_out, y)

                log_info(f"Saved {split_name} arrays: {x_out} ({X.shape}), {y_out} ({y.shape})")
                results[split_name] = {"X_shape": X.shape, "y_shape": y.shape, "saved": True}

            log_info("Data preprocessing completed successfully.")
            return results
        except CustomException:
            raise
        except Exception as e:
            log_error(f"Data preprocessing failed: {e}")
            raise CustomException("Data preprocessing failed", sys)


if __name__ == "__main__":
    dp = DataPreprocessing()
    out = dp.run()
    print(out)
