import numpy as np
import cv2


def color_map(mask):
    """
    Convert class mask → RGB image

    0 → background (black)
    1 → cloud (white)
    2 → shadow (blue)
    """

    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    colored[mask == 1] = [255, 255, 255]
    colored[mask == 2] = [0, 0, 255]

    return colored


def save_prediction(mask, path):
    colored = color_map(mask)
    cv2.imwrite(path, colored)


def overlay_prediction(image, mask, alpha=0.5):
    """
    Overlay mask on image for visualization
    """

    colored_mask = color_map(mask)

    image = image.astype(np.uint8)
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    return overlay