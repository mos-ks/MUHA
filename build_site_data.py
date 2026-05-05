"""Generate the annotations bundle the project page consumes.

For each of the 97 experiment images we collect three annotation layers:

* ``kitti``        – original KITTI labels read from
                     ``data/ground_truth/labels/<id>.txt``.
* ``predictions``  – probabilistic-EfficientDet predictions read from
                     ``data/detector_predictions.txt`` (carries the
                     per-coordinate aleatoric localization uncertainty).
* ``relabeled``    – the authors' corrected gold-standard read from
                     ``data/relabeled_ground_truth/labels/<id>.txt``.

The output is ``docs/data/annotations.json`` consumed by the static
viewer at ``docs/index.html``.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
GT_IMAGES = ROOT / "data" / "ground_truth" / "images"
GT_LABELS = ROOT / "data" / "ground_truth" / "labels"
RELABELED = ROOT / "data" / "relabeled_ground_truth" / "labels"
PREDICTIONS_FILE = ROOT / "data" / "detector_predictions.txt"
OUTPUT = ROOT / "docs" / "data" / "annotations.json"

# Class indices used in the prediction file are 1-based; index 5 is "Misc"
# in the 7-class scheme used by the paper (no Person_sitting).
CLASS_NAMES = {1: "Car", 2: "Van", 3: "Truck", 4: "Pedestrian",
               5: "Misc", 6: "Cyclist", 7: "Tram"}


def _parse_kitti(label_path: Path):
    """Parse a KITTI-format label file into a list of {cls, bbox}."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 8 or parts[0] == "DontCare":
            continue
        x1, y1, x2, y2 = (float(p) for p in parts[4:8])
        boxes.append({"cls": parts[0], "bbox": [x1, y1, x2, y2]})
    return boxes


def _parse_predictions():
    """Group every detector prediction by image id, keeping uncertainty."""
    by_image = defaultdict(list)
    if not PREDICTIONS_FILE.exists():
        return by_image
    for line in PREDICTIONS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = ast.literal_eval(line)
        # The prediction file stores boxes as [y1, x1, y2, x2]; flip to xyxy.
        y1, x1, y2, x2 = d["bbox"][:4]
        cls_idx = int(d.get("class", 0))
        cls = CLASS_NAMES.get(cls_idx, str(cls_idx))
        unc = d.get("rel_iso_perclscoo_albox", [0, 0, 0, 0])
        by_image[d["image_name"].split(".")[0]].append({
            "cls": cls,
            "bbox": [x1, y1, x2, y2],
            "score": float(d.get("score", d.get("det_score", 1.0))),
            "uncertainty": [float(v) for v in unc],
            "uncertainty_mean": float(sum(unc) / max(len(unc), 1)),
        })
    return by_image


def _image_size(path: Path):
    with Image.open(path) as im:
        return im.size  # (w, h)


def main():
    images = sorted(p.stem for p in GT_IMAGES.glob("*.png"))
    predictions = _parse_predictions()

    bundle = []
    for img_id in images:
        png = GT_IMAGES / f"{img_id}.png"
        w, h = _image_size(png)
        bundle.append({
            "id": img_id,
            "image": f"images/{img_id}.png",
            "width": w,
            "height": h,
            "kitti": _parse_kitti(GT_LABELS / f"{img_id}.txt"),
            "relabeled": _parse_kitti(RELABELED / f"{img_id}.txt"),
            "predictions": predictions.get(img_id, []),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(bundle, separators=(",", ":")))
    print(f"Wrote {len(bundle)} images to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
