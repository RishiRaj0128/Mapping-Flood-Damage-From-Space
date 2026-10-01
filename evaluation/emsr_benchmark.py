"""Benchmark evaluation against Copernicus EMS EMSR927 products.

Isolated from production inference code. Used only for offline validation and accuracy reporting.
"""



def calculate_emsr927_metrics(predicted_mask, reference_emsr_mask) -> dict[str, float]:
    """Calculates IoU, Precision, Recall, and F1 between pipeline predictions and EMSR927."""
    # Placeholder for offline benchmarking metrics
    return {
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
