from src.components.model_evaluation import ModelEvaluation

if __name__ == "__main__":
    evaluator = ModelEvaluation()
    out = evaluator.evaluate()
    print(out)
