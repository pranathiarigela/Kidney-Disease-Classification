from utils.common import create_dir, save_numpy, load_numpy, read_image
import numpy as np

# test directory
create_dir("test_dir")

# test numpy
arr = np.array([1,2,3])
save_numpy("test_dir/test.npy", arr)
loaded = load_numpy("test_dir/test.npy")
print("Numpy OK:", loaded)

# image test (use any image path)
# comment if no image available
# img = read_image("data/raw/normal/example.jpg")
# print("Image size:", img.size)
