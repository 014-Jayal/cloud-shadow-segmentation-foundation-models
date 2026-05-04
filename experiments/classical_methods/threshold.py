import os
import cv2
import numpy as np
from tqdm import tqdm


def apply_threshold(image):
    """
    Simple rule-based segmentation:
    - Cloud → high intensity
    - Shadow → low intensity
    - Background → everything else
    """

    # convert to grayscale for intensity-based thresholding
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # normalize
    gray = gray.astype(np.float32) / 255.0

    mask = np.zeros_like(gray, dtype=np.uint8)

    # cloud (bright regions)
    mask[gray > 0.7] = 1

    # shadow (dark regions)
    mask[gray < 0.2] = 2

    return mask


def load_image(path):
    if path.endswith(".tif"):
        import rasterio
        with rasterio.open(path) as src:
            img = src.read()
            img = np.transpose(img, (1, 2, 0))
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_mask(mask, path):
    """
    Save mask as grayscale:
    0,1,2 values preserved
    """
    cv2.imwrite(path, mask.astype(np.uint8))


def run_threshold(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))

    for file in tqdm(files, desc="Thresholding"):
        img_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)

        image = load_image(img_path)

        mask = apply_threshold(image)

        save_mask(mask, out_path)


if __name__ == "__main__":
    run_threshold(
        input_dir="data/images",
        output_dir="results/threshold"
    )