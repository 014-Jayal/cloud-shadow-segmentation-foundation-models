import numpy as np
import torch


def generate_patches(image, patch_size=224, stride=112):
    """
    Split large image into overlapping patches

    image: (H, W, C)
    returns: list of (patch, i, j)
    """
    H, W = image.shape[:2]
    patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append((patch, i, j))

    return patches


def reconstruct(preds, coords, image_shape, patch_size=224):
    """
    Reconstruct full prediction map from patches

    preds: list of (C, H, W)
    coords: list of (i, j)
    image_shape: (H, W)
    """
    H, W = image_shape
    C = preds[0].shape[0]

    output = np.zeros((C, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for pred, (i, j) in zip(preds, coords):
        output[:, i:i + patch_size, j:j + patch_size] += pred
        count_map[i:i + patch_size, j:j + patch_size] += 1.0

    count_map[count_map == 0] = 1.0
    output = output / count_map

    return output


def sliding_window_inference(model, image, device="cuda",
                             patch_size=224, stride=112):
    """
    Run inference on large image

    image: numpy array (H, W, C)
    returns: predicted class map (H, W)
    """

    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    patches = generate_patches(image, patch_size, stride)

    preds = []
    coords = []

    with torch.no_grad():
        for patch, i, j in patches:
            patch_tensor = torch.tensor(patch, dtype=torch.float32)
            patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0)
            patch_tensor = patch_tensor.to(device)

            output = model(patch_tensor)  # (1, C, H, W)
            output = torch.softmax(output, dim=1)

            preds.append(output.squeeze(0).cpu().numpy())
            coords.append((i, j))

    full_pred = reconstruct(preds, coords, image.shape[:2], patch_size)

    # final class prediction
    full_pred = np.argmax(full_pred, axis=0)

    return full_pred