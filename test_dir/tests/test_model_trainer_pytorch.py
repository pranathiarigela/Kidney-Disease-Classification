# tests/test_model_trainer_pytorch.py
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    trainer = ModelTrainer(epochs=5, batch_size=16, use_transfer_learning=True)
    out = trainer.run()
    print("Result:", out)
