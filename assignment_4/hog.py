import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, exposure
from skimage.feature import hog
from scipy import ndimage as ndi
import glob 

image_paths = glob.glob('pp_data/augmented/*/*.jpg') + glob.glob('pp_data/normalized/*/*.jpg')

pixels = 8
cells = 2
orientations = 50

hog_features_list = []

for i, image_path in enumerate(image_paths):

    image = io.imread(image_path)
    grayscale_image = color.rgb2gray(image)

    gradient_x = ndi.sobel(grayscale_image, axis=1)
    gradient_y = ndi.sobel(grayscale_image, axis=0)

    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    hog_features, hog_image = hog(grayscale_image, 
                                  pixels_per_cell=(pixels, pixels), 
                                  cells_per_block=(cells, cells), 
                                  orientations=orientations, 
                                  visualize=True, 
                                  block_norm='L2-Hys')
    
    hog_features_list.append(hog_features)
    
hog_features_array = np.array(hog_features_list)
print(f"HOG features array shape: {hog_features_array.shape}")
np.save("features/hog_features.npy", hog_features_array)