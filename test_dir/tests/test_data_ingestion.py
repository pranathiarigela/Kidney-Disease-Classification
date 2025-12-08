# tests/test_data_ingestion.py
from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    di = DataIngestion()
    out = di.run()
    print(out)
