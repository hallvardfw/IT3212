import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import local_binary_pattern
import numpy as np
import glob

image_paths = glob.glob('pp_data/augmented/*/*.jpg') + glob.glob('pp_data/normalized/*/*.jpg')

radius = 1
n_points = 8 * radius

lbp_features_list = []

for i, image_path in enumerate(image_paths):
    image = io.imread(image_path)
    grayscale_image = color.rgb2gray(image)

    grayscale_image = (grayscale_image * 255).astype(np.uint8)

    lbp = local_binary_pattern(grayscale_image, n_points, radius, method='uniform')

    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    lbp_features_list.append(lbp_hist)

lbp_features_array = np.array(lbp_features_list)
print(f"LBP features array shape: {lbp_features_array.shape}")

np.save("features/lbp_features.npy", lbp_features_array)