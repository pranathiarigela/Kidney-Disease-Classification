# tests/test_prediction_pipeline.py
from src.pipeline.prediction_pipeline import PredictionPipeline
from pathlib import Path
import glob

if __name__ == "__main__":
    pp = PredictionPipeline()
    files = glob.glob(str(Path("data/processed/test") / "*" / "*"))
    if not files:
        print("No test images found. Put an image under data/processed/test/<class>/ or pass a path.")
    else:
        print("Using:", files[0])
        res = pp.predict(files[0])
        print(res)
