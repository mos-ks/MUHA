"""Convert ``data/relabeled_ground_truth.txt`` into per-image KITTI-format
label files under ``data/relabeled_ground_truth/labels/``.

The master file is one Python-literal dict per line. Each record carries
the detector's prediction (``bbox``) and, when the prediction matched a
ground-truth box, the corrected ground-truth coords (``gt_bbox``). Score
== 1.0 marks records that the authors *added* during relabeling — these
are GT-only entries with no underlying detector prediction. Both ``bbox``
and ``gt_bbox`` are stored in ``[y1, x1, y2, x2]`` order in the source
file; KITTI's native order is ``[x1, y1, x2, y2]``, hence the swap below.
"""

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "relabeled_ground_truth.txt"
DST = ROOT / "data" / "relabeled_ground_truth" / "labels"

# Class indices in the prediction file are 1-based; index 5 is "Misc" in
# the 7-class scheme used by the paper (no Person_sitting).
CLASS_NAMES = {1: "Car", 2: "Van", 3: "Truck", 4: "Pedestrian",
               5: "Misc", 6: "Cyclist", 7: "Tram"}


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    per_image: dict[str, list[tuple]] = defaultdict(list)

    for line in SRC.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = ast.literal_eval(line)
        img_id = d["image_name"].split(".")[0]
        score = d.get("score", d.get("det_score", 0))

        if score < 1.0 and d.get("gt_bbox"):
            # Prediction matched a (possibly corrected) ground-truth box.
            y1, x1, y2, x2 = d["gt_bbox"]
            cls = int(d.get("gt_class", d.get("class", 1)))
        else:
            # Manually added GT box (no detector match).
            y1, x1, y2, x2 = d["bbox"]
            cls = int(d.get("class", 1))

        per_image[img_id].append((CLASS_NAMES.get(cls, "Misc"), x1, y1, x2, y2))

    for img_id, boxes in per_image.items():
        lines = [
            f"{cls} 0.00 0 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"-1 -1 -1 -1000 -1000 -1000 -10"
            for cls, x1, y1, x2, y2 in boxes
        ]
        (DST / f"{img_id}.txt").write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(per_image)} files, {sum(len(v) for v in per_image.values())} boxes.")


if __name__ == "__main__":
    main()
