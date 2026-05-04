import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier


def extract_features(image):
    """
    Extract pixel-wise features:
    - RGB / multispectral values
    - spatial coordinates
    """

    h, w, c = image.shape

    pixels = image.reshape(-1, c).astype(np.float32) / 255.0

    # spatial coordinates
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([xs, ys], axis=-1).reshape(-1, 2)

    # normalize coords
    coords = coords / np.array([[w, h]])

    features = np.concatenate([pixels, coords], axis=1)

    return features


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


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask.astype(np.uint8)  # 0,1,2


def train_rf(image_paths, mask_paths, sample_size=300000):
    """
    Train RF with sampling (important for speed)
    """

    X = []
    y = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        image = load_image(img_path)
        mask = load_mask(mask_path)

        features = extract_features(image)
        labels = mask.reshape(-1)

        X.append(features)
        y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # sampling (critical for RF performance)
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    print(f"[INFO] Training RF on {X.shape[0]} samples")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X, y)

    return model


def predict_rf(model, image):
    h, w, _ = image.shape

    features = extract_features(image)
    preds = model.predict(features)

    mask = preds.reshape(h, w).astype(np.uint8)

    return mask


def run_rf(train_img_dir, train_mask_dir, test_img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_imgs = sorted(os.listdir(train_img_dir))
    train_masks = sorted(os.listdir(train_mask_dir))

    train_img_paths = [os.path.join(train_img_dir, f) for f in train_imgs]
    train_mask_paths = [os.path.join(train_mask_dir, f) for f in train_masks]

    # train
    model = train_rf(train_img_paths, train_mask_paths)

    # inference
    test_files = sorted(os.listdir(test_img_dir))

    for file in tqdm(test_files, desc="RF Inference"):
        img_path = os.path.join(test_img_dir, file)
        out_path = os.path.join(output_dir, file)

        image = load_image(img_path)

        mask = predict_rf(model, image)

        cv2.imwrite(out_path, mask)


if __name__ == "__main__":
    run_rf(
        train_img_dir="data/train/images",
        train_mask_dir="data/train/masks",
        test_img_dir="data/test/images",
        output_dir="results/rf"
    )