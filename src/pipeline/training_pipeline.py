# src/pipeline/training_pipeline.py
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from utils.logger import log_info, log_error
from utils.exception import CustomException

# import components (assumes these files exist as provided earlier)
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    def __init__(self,
                 raw_data_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 models_dir: str = "models",
                 batch_size: int = 32,
                 epochs: int = 20,
                 use_transfer_learning: bool = True,
                 resume_training: bool = False):
        try:
            self.raw_data_dir = Path(raw_data_dir)
            self.processed_dir = Path(processed_dir)
            self.models_dir = Path(models_dir)
            self.batch_size = batch_size
            self.epochs = epochs
            self.use_tl = use_transfer_learning
            self.resume_training = resume_training

            # ensure folders
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(parents=True, exist_ok=True)

            log_info(f"TrainingPipeline initialized at {datetime.now().isoformat()}")
        except Exception as e:
            log_error(f"Failed to initialize pipeline: {e}")
            raise CustomException("Failed to initialize TrainingPipeline", sys)

    def run_ingestion(self):
        try:
            log_info("Starting data ingestion...")
            di = DataIngestion(raw_data_dir=str(self.raw_data_dir),
                               processed_dir=str(self.processed_dir))
            result = di.run()
            log_info(f"Data ingestion finished: {result}")
            return result
        except CustomException:
            raise
        except Exception as e:
            log_error(f"Ingestion step failed: {e}")
            raise CustomException("Ingestion failed", sys)

    def run_preprocessing(self):
        try:
            log_info("Starting data preprocessing...")
            dp = DataPreprocessing(processed_dir=str(self.processed_dir))
            result = dp.run()
            log_info(f"Data preprocessing finished: {result}")
            return result
        except CustomException:
            raise
        except Exception as e:
            log_error(f"Preprocessing step failed: {e}")
            raise CustomException("Preprocessing failed", sys)

    def run_training(self):
        try:
            log_info("Starting model training...")
            trainer = ModelTrainer(processed_dir=str(self.processed_dir),
                                   models_dir=str(self.models_dir),
                                   batch_size=self.batch_size,
                                   epochs=self.epochs,
                                   use_transfer_learning=self.use_tl)
            result = trainer.run()
            log_info(f"Model training finished: {result}")
            return result
        except CustomException:
            raise
        except Exception as e:
            log_error(f"Training step failed: {e}")
            raise CustomException("Training failed", sys)

    def run_evaluation(self):
        try:
            log_info("Starting model evaluation...")
            evaluator = ModelEvaluation(processed_dir=str(self.processed_dir),
                                        models_dir=str(self.models_dir))
            result = evaluator.evaluate()
            log_info(f"Model evaluation finished: {result}")
            return result
        except CustomException:
            raise
        except Exception as e:
            log_error(f"Evaluation step failed: {e}")
            raise CustomException("Evaluation failed", sys)

    def run_pipeline(self,
                     do_ingest: bool = True,
                     do_preprocess: bool = True,
                     do_train: bool = True,
                     do_eval: bool = True):
        summary = {
            "timestamp": datetime.now().isoformat(),
            "ingestion": None,
            "preprocessing": None,
            "training": None,
            "evaluation": None,
            "status": "started"
        }
        try:
            if do_ingest:
                summary["ingestion"] = self.run_ingestion()
            else:
                log_info("Skipping ingestion step (flag disabled)")

            if do_preprocess:
                summary["preprocessing"] = self.run_preprocessing()
            else:
                log_info("Skipping preprocessing step (flag disabled)")

            if do_train:
                summary["training"] = self.run_training()
            else:
                log_info("Skipping training step (flag disabled)")

            if do_eval:
                summary["evaluation"] = self.run_evaluation()
            else:
                log_info("Skipping evaluation step (flag disabled)")

            summary["status"] = "completed"
            log_info("Pipeline completed successfully.")
        except CustomException as ce:
            summary["status"] = "failed"
            summary["error"] = str(ce)
            log_error(f"Pipeline failed: {ce}")
            raise
        except Exception as e:
            summary["status"] = "failed"
            summary["error"] = str(e)
            log_error(f"Unexpected pipeline failure: {e}")
            raise CustomException("Pipeline run failed", sys)
        finally:
            # persist summary
            try:
                out_path = self.models_dir / "pipeline_run_summary.json"
                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)
                log_info(f"Saved pipeline summary to {out_path}")
            except Exception as e:
                log_error(f"Failed to save pipeline summary: {e}")

        return summary


def parse_args():
    p = argparse.ArgumentParser(description="Run end-to-end training pipeline")
    p.add_argument("--no-ingest", action="store_true", help="Skip data ingestion")
    p.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing")
    p.add_argument("--no-train", action="store_true", help="Skip training")
    p.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    p.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--no-tl", action="store_true", help="Disable transfer learning (use small CNN)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = TrainingPipeline(batch_size=args.batch_size,
                                epochs=args.epochs,
                                use_transfer_learning=not args.no_tl)
    # decide steps
    do_ingest = not args.no_ingest
    do_preprocess = not args.no_preprocess
    do_train = not args.no_train
    do_eval = not args.no_eval

    pipeline.run_pipeline(
        do_ingest=do_ingest,
        do_preprocess=do_preprocess,
        do_train=do_train,
        do_eval=do_eval
    )
