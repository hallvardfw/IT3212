import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import local_binary_pattern
import numpy as np

image_paths = ['data/portrait.jpg', 'data/landscape.jpg']

for i, image_path in enumerate(image_paths):
    image = io.imread(image_path)
    grayscale_image = color.rgb2gray(image)

    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(grayscale_image, n_points, radius, method='uniform')

    n_bins = int(lbp.max() + 1)
    lbp_hist, bin_edges = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    fig, ax1 = plt.subplots()

    ax1.imshow(lbp, cmap='gray')
    ax1.set_title('LBP Image')

    plt.tight_layout()
    plt.savefig(f'lbp{i}.jpg', bbox_inches='tight')