import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def load_image(path=None):
    """Loads an image or generates a synthetic one if path is None."""
    if path:
        # Load as Grayscale (0 flag is crucial for MRA)
        img = cv2.imread(path, 0)
        if img is None:
            raise ValueError(f"Could not load image at {path}")
    else:
        print("[INFO] No path provided. Generating synthetic image for demo...")
        img = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, -1)
        cv2.circle(img, (256, 256), 100, 0, -1)
        # Add noise
        noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    return img

def visualize_decomposition(coeffs):
    """
    Plots the Approximations and Details for all levels.
    coeffs structure: [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
    """
    level_count = len(coeffs) - 1

    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Wavelet Decomposition (Levels 1 to {level_count})", fontsize=16)

    # 1. Plot Approximation (Lowest Level)
    cA = coeffs[0]
    ax = fig.add_subplot(level_count + 1, 4, 1)
    ax.imshow(cA, cmap='gray')
    ax.set_title(f"Approx (LL{level_count})")
    ax.axis('off')

    # 2. Loop through details
    # coeffs[1] is Level n details, coeffs[2] is Level n-1...
    for i, details in enumerate(coeffs[1:]):
        level_index = level_count - i
        (cH, cV, cD) = details

        # Determine plot row (starts after approximation)
        row = i + 1

        # Horizontal Details (LH)
        ax1 = fig.add_subplot(level_count + 1, 4, row * 4 - 2)
        ax1.imshow(cH, cmap='gray')
        ax1.set_title(f"Horizontal (LH{level_index})")
        ax1.axis('off')

        # Vertical Details (HL)
        ax2 = fig.add_subplot(level_count + 1, 4, row * 4 - 1)
        ax2.imshow(cV, cmap='gray')
        ax2.set_title(f"Vertical (HL{level_index})")
        ax2.axis('off')

        # Diagonal Details (HH)
        ax3 = fig.add_subplot(level_count + 1, 4, row * 4)
        ax3.imshow(cD, cmap='gray')
        ax3.set_title(f"Diagonal (HH{level_index})")
        ax3.axis('off')

    plt.tight_layout()
    plt.show()

def extract_features(coeffs):
    """Extracts statistical features from the coefficients."""
    features = {}

    # Helper for Entropy
    def calc_entropy(matrix):
        p = np.abs(matrix.flatten())
        p = p / (np.sum(p) + 1e-10)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    # Process Approximation
    cA = coeffs[0]
    level = len(coeffs) - 1
    features[f'LL{level}_Energy'] = np.sum(cA**2)
    features[f'LL{level}_Mean'] = np.mean(cA)

    # Process Details
    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        curr_lvl = level - i

        # Loop over the tuple of details
        for band, name in zip([cH, cV, cD], ['LH', 'HL', 'HH']):
            features[f'{name}{curr_lvl}_Energy'] = np.sum(band**2)
            features[f'{name}{curr_lvl}_Entropy'] = calc_entropy(band)
            features[f'{name}{curr_lvl}_StdDev'] = np.std(band)

    return pd.DataFrame([features]).T

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Correct path for Google Colab environment
    image_path = "/content/porsche-911-turbo-s-3840x2160-23991.jpg"

    try:
        # 1. Load Image
        original_img = load_image(image_path)

        # 2. Perform Wavelet Decomposition (MRA)
        coeffs = pywt.wavedec2(original_img, 'db2', level=3)

        # 3. SET UP PLOTS (Don't show yet)

        # Figure 1: Original Image
        plt.figure(figsize=(6,6))
        plt.imshow(original_img, cmap='gray')
        plt.title("Original Input Image")
        plt.axis('off')

        # Figure 2: MRA Decomposition
        visualize_decomposition(coeffs)

        # 4. Extract and Print Features
        print("\nExtracting Features...")
        feats = extract_features(coeffs)
        print("="*40)
        print("FINAL FEATURE VECTOR")
        print("="*40)
        print(feats)

    except Exception as e:
        print(f"Error: {e}")