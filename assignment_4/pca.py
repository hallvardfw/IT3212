import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

hog_features_array = np.load("features/hog_features.npy")
lbp_features_array = np.load("features/lbp_features.npy")

print(f"HOG features shape: {hog_features_array.shape}")
print(f"LBP features shape: {lbp_features_array.shape}")

combined_features = np.hstack((hog_features_array, lbp_features_array))
print(f"Combined features shape: {combined_features.shape}")

scaler = StandardScaler()
standardized_features = scaler.fit_transform(combined_features)
print("Features standardized.")

pca = PCA(n_components=0.95)
reduced_features = pca.fit_transform(standardized_features)
print(f"Reduced features shape: {reduced_features.shape}")

np.save("features/reduced_features.npy", reduced_features)