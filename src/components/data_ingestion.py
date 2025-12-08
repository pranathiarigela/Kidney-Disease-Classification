import os
import shutil
import json
from glob import glob
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split

import sys
from utils.logger import log_info, log_error
from utils.exception import CustomException

class DataIngestion:
    def __init__(self,
                 raw_data_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42):
        try:
            self.raw_data_dir = Path(raw_data_dir)
            self.processed_dir = Path(processed_dir)
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.test_ratio = test_ratio
            self.random_state = random_state

            # sanity checks
            if not self.raw_data_dir.exists():
                raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")

            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

            # create target directories
            (self.processed_dir / "train").mkdir(parents=True, exist_ok=True)
            (self.processed_dir / "val").mkdir(parents=True, exist_ok=True)
            (self.processed_dir / "test").mkdir(parents=True, exist_ok=True)

            log_info(f"Initialized DataIngestion: raw={self.raw_data_dir}, processed={self.processed_dir}")
        except Exception as e:
            log_error(f"Failed initializing DataIngestion: {e}")
            raise CustomException("Failed to initialize DataIngestion", sys)

    def _gather_files(self) -> Dict[str, List[str]]:
        """
        Returns a dict: {class_name: [file1, file2, ...], ...}
        Only collects common image extensions.
        """
        try:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
            class_files = {}
            for class_dir in sorted(self.raw_data_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                files = []
                for ext in exts:
                    files.extend(map(str, class_dir.glob(ext)))
                if files:
                    class_files[class_dir.name] = files
                    log_info(f"Found {len(files)} files for class '{class_dir.name}'")
            if not class_files:
                raise FileNotFoundError("No image files found under raw data directory.")
            return class_files
        except Exception as e:
            log_error(f"Error gathering files: {e}")
            raise CustomException("Error gathering files", sys)

    def _split_list(self, items: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Splits a list into train, val, test based on ratios.
        """
        try:
            train_and_temp, test = train_test_split(items, test_size=self.test_ratio, random_state=self.random_state)
            # compute relative val size with respect to train_and_temp
            val_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
            train, val = train_test_split(train_and_temp, test_size=val_relative, random_state=self.random_state)
            return train, val, test
        except Exception as e:
            log_error(f"Error in splitting list: {e}")
            raise CustomException("Error splitting list", sys)

    def _copy_files(self, files_with_labels: List[Tuple[str, int]], split_name: str):
        """
        Copy files into processed/<split>/<class_name> folders and return list of destination filepaths.
        files_with_labels: list of (src_path, label)
        split_name: 'train'/'val'/'test'
        """
        try:
            dest_paths = []
            for src_path, label in files_with_labels:
                src = Path(src_path)
                class_name = src.parent.name
                dest_dir = self.processed_dir / split_name / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / src.name
                shutil.copy2(src, dest_file)
                dest_paths.append((str(dest_file), label))
            log_info(f"Copied {len(dest_paths)} files to {split_name}")
            return dest_paths
        except Exception as e:
            log_error(f"Error copying files for {split_name}: {e}")
            raise CustomException(f"Error copying files for {split_name}", sys)

    def run(self):
        """
        Main entry to run data ingestion.
        Outputs:
          - data/processed/train/<class>/*
          - data/processed/val/<class>/*
          - data/processed/test/<class>/*
          - data/processed/label_map.json
          - data/processed/train.csv / val.csv / test.csv
        """
        try:
            class_files = self._gather_files()

            # create label map
            classes = sorted(class_files.keys())
            label_map = {cls: idx for idx, cls in enumerate(classes)}
            label_map_path = self.processed_dir / "label_map.json"
            with open(label_map_path, "w") as f:
                json.dump(label_map, f, indent=2)
            log_info(f"Saved label map with {len(label_map)} classes to {label_map_path}")

            # collect splits
            train_records = []
            val_records = []
            test_records = []

            for cls, files in class_files.items():
                train, val, test = self._split_list(files)
                cls_label = label_map[cls]
                train_records += [(fp, cls_label) for fp in train]
                val_records += [(fp, cls_label) for fp in val]
                test_records += [(fp, cls_label) for fp in test]

            # copy files to processed dir
            train_dest = self._copy_files(train_records, "train")
            val_dest = self._copy_files(val_records, "val")
            test_dest = self._copy_files(test_records, "test")

            # save CSV lists
            def save_csv(records: List[Tuple[str, int]], fname: str):
                import csv
                path = self.processed_dir / fname
                with open(path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["filepath", "label"])
                    for fp, label in records:
                        writer.writerow([fp, label])
                log_info(f"Saved {len(records)} records to {path}")

            save_csv(train_dest, "train.csv")
            save_csv(val_dest, "val.csv")
            save_csv(test_dest, "test.csv")

            log_info("Data ingestion completed successfully.")
            return {
                "train_count": len(train_dest),
                "val_count": len(val_dest),
                "test_count": len(test_dest),
                "label_map_path": str(label_map_path)
            }
        except Exception as e:
            log_error(f"Data ingestion failed: {e}")
            raise CustomException("Data ingestion failed", sys)


if __name__ == "__main__":
    # simple runner for manual testing
    di = DataIngestion()
    res = di.run()
    print("Ingestion result:", res)
