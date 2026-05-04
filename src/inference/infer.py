import os
import argparse
import numpy as np
import torch
import cv2
import rasterio

from src.models.prithvi_adapter import PrithviAdapter, load_prithvi_backbone
from src.models.decoder import SimpleDecoder, CloudSegmentationModel
from src.inference.sliding_window import sliding_window_inference


def load_image(path):
    """
    Supports .tif and normal images
    """
    if path.endswith(".tif"):
        with rasterio.open(path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
    else:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0
    return image


def color_map(mask):
    """
    Convert class mask to RGB visualization
    0: background → black
    1: cloud → white
    2: shadow → blue
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    colored[mask == 1] = [255, 255, 255]   # cloud
    colored[mask == 2] = [0, 0, 255]       # shadow

    return colored


def save_output(mask, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    colored = color_map(mask)
    cv2.imwrite(path, colored)


def build_model(checkpoint_path, device):
    """
    Build full model
    """

    backbone = load_prithvi_backbone()
    backbone = PrithviAdapter(backbone)

    decoder = SimpleDecoder(in_channels=768, num_classes=3)

    model = CloudSegmentationModel(backbone, decoder)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    return model


def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading model...")
    model = build_model(args.checkpoint, device)

    print("[INFO] Loading image...")
    image = load_image(args.input)

    print("[INFO] Running inference...")
    pred = sliding_window_inference(
        model,
        image,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride
    )

    print("[INFO] Saving result...")
    save_output(pred, args.output)

    print("[DONE] Inference complete")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="results/output.png")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)