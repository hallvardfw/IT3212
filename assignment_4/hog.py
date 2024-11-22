import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, exposure
from skimage.feature import hog
from scipy import ndimage as ndi

image_paths = ['data/suv/PIC_11.jpg', 'data/suv/PIC_5.jpg', 'data/motorcycle/PIC_157.jpg']

pixels = 8
cells = 2
orientations = 50

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
    
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    fig, ax1 = plt.subplots()

    ax1.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax1.set_title('HOG Features')

    plt.tight_layout()
    plt.savefig(f'hog{i}.jpg', bbox_inches='tight')