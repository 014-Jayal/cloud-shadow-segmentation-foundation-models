import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


def apply_kmeans(image, k=3):
    """
    K-Means clustering for multi-class segmentation
    """

    h, w, c = image.shape

    # reshape pixels
    pixels = image.reshape(-1, c).astype(np.float32)

    # normalize
    pixels = pixels / 255.0

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    labels = labels.reshape(h, w)

    # determine cluster types based on intensity
    cluster_means = []

    for i in range(k):
        cluster_pixels = pixels[labels.reshape(-1) == i]
        if len(cluster_pixels) == 0:
            cluster_means.append(0)
        else:
            cluster_means.append(np.mean(cluster_pixels))

    cluster_means = np.array(cluster_means)

    # assign classes
    sorted_idx = np.argsort(cluster_means)

    mask = np.zeros((h, w), dtype=np.uint8)

    # lowest intensity → shadow
    mask[labels == sorted_idx[0]] = 2

    # middle → background
    mask[labels == sorted_idx[1]] = 0

    # highest → cloud
    mask[labels == sorted_idx[2]] = 1

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
    cv2.imwrite(path, mask.astype(np.uint8))


def run_kmeans(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))

    for file in tqdm(files, desc="KMeans"):
        img_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)

        image = load_image(img_path)

        mask = apply_kmeans(image)

        save_mask(mask, out_path)


if __name__ == "__main__":
    run_kmeans(
        input_dir="data/images",
        output_dir="results/kmeans"
    )