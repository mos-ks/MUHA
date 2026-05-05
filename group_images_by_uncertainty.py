"""Group KITTI validation images into low/mid/high difficulty bins.

Difficulty is defined by the per-image average aleatoric localization
uncertainty (rel_iso_perclscoo_albox) emitted by the probabilistic
EfficientDet-D0 used in the paper. Images are split at the 33rd and 67th
percentiles, matching the binning described in the Methods section.

Inputs : data/detector_predictions.txt, data/ground_truth/images/
Outputs: data/{low,mid,high}_uncertainty.txt
"""

import os
import numpy as np

PREDICTIONS_FILE = "./data/detector_predictions.txt"
IMAGE_DIR = "./data/ground_truth/images"
UNCERTAINTY_KEY = "rel_iso_perclscoo_albox"


def load_uncertainties(predictions_file: str) -> dict[str, float]:
    """Return mean per-image uncertainty across all detections in the file."""
    per_image = {}
    with open(predictions_file, "r") as f:
        for line in f:
            data = eval(line.strip())
            img = data.get("image_name", "").split(".")[0]
            unc = data.get(UNCERTAINTY_KEY)
            if img and unc is not None:
                per_image.setdefault(img, []).append(np.mean(unc))
    return {img: float(np.mean(vals)) for img, vals in per_image.items()}


def list_images(image_dir: str) -> list[str]:
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def main() -> None:
    avg_unc = load_uncertainties(PREDICTIONS_FILE)
    images = [img for img in list_images(IMAGE_DIR) if img in avg_unc]
    if not images:
        print("No images with uncertainty values found.")
        return

    values = np.array([avg_unc[img] for img in images])
    low_q, high_q = np.quantile(values, [1 / 3, 2 / 3])

    low, mid, high = [], [], []
    for img in images:
        v = avg_unc[img]
        (low if v <= low_q else high if v > high_q else mid).append(img)

    for name, ids in (("low", low), ("mid", mid), ("high", high)):
        with open(f"./data/{name}_uncertainty.txt", "w") as f:
            f.write("\n".join(sorted(ids)) + "\n")

    print(f"Wrote {len(low)} low, {len(mid)} mid, {len(high)} high.")


if __name__ == "__main__":
    main()
