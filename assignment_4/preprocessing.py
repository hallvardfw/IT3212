import cv2
import tensorflow as tf
import os
import numpy as np
import glob

def resize_images(input_dir, output_dir, size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, size)
                cv2.imwrite(os.path.join(output_category_path, img_name), resized_img)
def convert_to_grayscale(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(output_category_path, img_name), gray_img)
def normalize_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                normalized_img = img / 255.0
                # Save the normalized image (convert to uint8 for saving if needed)
                cv2.imwrite(os.path.join(output_category_path, img_name), (normalized_img * 255).astype(np.uint8))
def augment_image(image_path:str, output_dir):
    paths = image_path.split('/')
    if len(paths) > 1:
        output_dir = f'{output_dir}/{paths[-2]}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    new_image = paths[-1]
    tf.keras.preprocessing.image.save_img(f'{output_dir}/{new_image}', image.numpy())
    

resize_images('assignment_4/data', 'assignment_4/pp_data/resize')
convert_to_grayscale('assignment_4/pp_data/resize', 'assignment_4/pp_data/grayscale')
normalize_images('assignment_4/pp_data/grayscale', 'assignment_4/pp_data/normalized')

categories = ['hatchback', 'motorcycle', 'pickup', 'sedan', 'suv']
file_paths = glob.glob('assignment_4/pp_data/normalized/*')

category_files = {category: glob.glob(f'{file_path}/*')[:10] for category, file_path in zip(categories, file_paths)}

for category, images in category_files.items():
    for image in images:
        augment_image(image, 'assignment_4/pp_data/augmented')