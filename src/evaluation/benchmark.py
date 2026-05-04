import os
import numpy as np
import cv2
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


def load_mask(path):
    """
    Load ground truth or prediction mask
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask.astype(np.int64)


def evaluate_model(pred_dir, gt_dir):
    """
    Evaluate a model given prediction folder and ground truth folder
    """

    files = sorted(os.listdir(gt_dir))

    metrics_list = []

    for file in tqdm(files, desc="Evaluating"):
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)

        if not os.path.exists(pred_path):
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        metrics = compute_metrics(pred, gt)
        metrics_list.append(metrics)

    # average metrics
    final_metrics = {}

    for key in metrics_list[0].keys():
        final_metrics[key] = np.mean([m[key] for m in metrics_list])

    return final_metrics


def print_results(results):
    """
    Print clean comparison table
    """

    print("\n" + "=" * 90)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'mIoU':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 90)

    for model, metrics in results.items():
        print(
            f"{model:<20} | "
            f"{metrics['accuracy']:.4f}     | "
            f"{metrics['mIoU']:.4f}     | "
            f"{metrics['f1']:.4f}     | "
            f"{metrics['precision']:.4f}     | "
            f"{metrics['recall']:.4f}"
        )

    print("=" * 90)


def run_benchmark(models_dict, gt_dir):
    """
    models_dict:
        {
            "model_name": "path/to/predictions"
        }
    """

    results = {}

    for model_name, pred_dir in models_dict.items():
        print(f"\nRunning evaluation for: {model_name}")
        results[model_name] = evaluate_model(pred_dir, gt_dir)

    print_results(results)

    return results


if __name__ == "__main__":
    models = {
        "threshold": "results/threshold",
        "kmeans": "results/kmeans",
        "random_forest": "results/rf",
        "svm": "results/svm",
        "cnn": "results/cnn",
        "prithvi": "results/prithvi"
    }

    gt_dir = "data/gt_masks"

    run_benchmark(models, gt_dir)