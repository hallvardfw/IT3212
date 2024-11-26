import os
import numpy as np

def generate_labels_from_two_folders(feature_array, folder1, folder2):
    labels = []
    class_names = []

    subfolders1 = sorted([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
    subfolders2 = sorted([d for d in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, d))])
    
    if subfolders1 != subfolders2:
        print(subfolders1)
        print(subfolders2)
        raise ValueError("The subfolder structures of the two folders do not match.")

    for label_idx, subfolder in enumerate(subfolders1):
        path1 = os.path.join(folder1, subfolder)
        path2 = os.path.join(folder2, subfolder)
        
        class_names.append(subfolder)

        num_images1 = len([f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))])
        num_images2 = len([f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))])
        
        total_images = num_images1 + num_images2
        labels.extend([label_idx] * total_images)

    labels = np.array(labels[:len(feature_array)])
    return labels, class_names

feature_array = np.load("features/reduced_features.npy")
folder1 = "pp_data/augmented"
folder2 = "pp_data/normalized"

labels, class_names = generate_labels_from_two_folders(feature_array, folder1, folder2)

print(f"Generated {len(labels)} labels for {len(class_names)} classes: {class_names}")

np.save("features/labels.npy", labels)
np.save("features/class_names.npy", class_names)