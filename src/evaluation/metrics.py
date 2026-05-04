import numpy as np


def compute_confusion_matrix(pred, gt, num_classes=3):
    """
    Compute confusion matrix
    """

    mask = (gt >= 0) & (gt < num_classes)
    conf_mat = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return conf_mat


def compute_metrics(pred, gt, num_classes=3):
    """
    Computes:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - mIoU
    """

    conf_mat = compute_confusion_matrix(pred, gt, num_classes)

    # overall accuracy
    accuracy = np.sum(np.diag(conf_mat)) / (np.sum(conf_mat) + 1e-6)

    precision = []
    recall = []
    f1 = []
    iou = []

    for cls in range(num_classes):
        tp = conf_mat[cls, cls]
        fp = conf_mat[:, cls].sum() - tp
        fn = conf_mat[cls, :].sum() - tp

        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1_score = 2 * prec * rec / (prec + rec + 1e-6)
        iou_score = tp / (tp + fp + fn + 1e-6)

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
        iou.append(iou_score)

    metrics = {
        "accuracy": accuracy,
        "precision": np.mean(precision),
        "recall": np.mean(recall),
        "f1": np.mean(f1),
        "mIoU": np.mean(iou)
    }

    return metrics


def print_metrics(metrics):
    print("\nEvaluation Results:")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"mIoU      : {metrics['mIoU']:.4f}")