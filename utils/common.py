import os
import numpy as np
from PIL import Image

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def save_numpy(path, arr):
    np.save(path, arr)

def load_numpy(path):
    return np.load(path)

def read_image(path):
    img = Image.open(path).convert("RGB")
    return img
