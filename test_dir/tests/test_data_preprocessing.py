# tests/test_data_preprocessing.py
from src.components.data_preprocessing import DataPreprocessing

if __name__ == "__main__":
    dp = DataPreprocessing(verbose=True)
    res = dp.run()
    print("Preprocessing result:", res)
