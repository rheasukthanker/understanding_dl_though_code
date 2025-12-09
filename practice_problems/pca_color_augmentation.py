


import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # Reshape image to (H*W, 3)
    original_shape = image.shape
    pixels = image.reshape(-1, 3).astype(np.float64)
    
    # Compute mean and center the data
    mean_rgb = np.mean(pixels, axis=0)
    centered_pixels = pixels - mean_rgb
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered_pixels.T)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute color distortion
    distortion = eigenvectors @ (alpha * np.sqrt(np.maximum(eigenvalues, 0)))
    
    # Apply distortion to all pixels
    augmented_pixels = pixels + distortion
    
    # Reshape back to original shape and clamp values
    augmented_image = augmented_pixels.reshape(original_shape)
    augmented_image = np.clip(augmented_image, 0, 255)
    
    return augmented_image