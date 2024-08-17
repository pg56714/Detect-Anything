import numpy as np
import cv2
from matplotlib import pyplot as plt


def dilate_mask(input_path, dilate_factor):
    """
    Reads a mask from a PNG file, dilates it using a square kernel of size 'dilate_factor',
    and plots the result.

    Parameters:
    input_path (str): The path to the PNG file containing the binary mask.
    dilate_factor (int): The size of the square kernel used for dilation.
    """
    # Read the mask image
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Image could not be loaded. Please check the file path.")

    # Ensure mask is binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Create the dilation kernel
    kernel = np.ones((dilate_factor, dilate_factor), np.uint8)

    # Dilate the mask
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Plot the original and dilated mask
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray", interpolation="nearest")
    plt.title("Original Mask")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(dilated_mask, cmap="gray", interpolation="nearest")
    plt.title("Dilated Mask")
    plt.axis("off")

    plt.show()


# Example usage:
dilate_mask("./imagetest.png", 5)
