# tests/inspect_prediction.py
import json
import numpy as np
from pathlib import Path
from src.pipeline.prediction_pipeline import PredictionPipeline
img_path = "C:\\Users\\apran\\Desktop\\KDC\\data\\raw\\Tumor\\Tumor- (7).jpg"
pp = PredictionPipeline()   # auto-detects device

print("Label map (idx->name):", pp.inv_label_map)

res = pp.predict(img_path)
print("Top-1:", res)

# try to get full probs
try:
    import torch
    import numpy as np
    from torch.nn import functional as F
    x = pp._preprocess(img_path)
    with torch.no_grad():
        out = pp.model(x)
        probs = F.softmax(out, dim=1).cpu().numpy().squeeze()
    print("Probs vector:", probs)
    topk = np.argsort(probs)[::-1][:5]
    for i in topk:
        print(f"{pp.inv_label_map.get(int(i))}: {probs[int(i)]:.4f}")
except Exception as e:
    print("Could not get full probs:", e)
