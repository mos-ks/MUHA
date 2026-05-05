# Data License and Attribution

This repository contains a small subset of the KITTI dataset together with
derived files (detector predictions, uncertainty bins, and re-annotated
ground-truth labels). This document explains what is licensed under what
and how to use it correctly.

## KITTI subset (`data/ground_truth/`)

The 136 PNG images under `data/ground_truth/images/` and the original
annotation files under `data/ground_truth/labels/` are taken verbatim from
the KITTI 2D Object Detection benchmark.

The KITTI dataset is published by Andreas Geiger, Philip Lenz, and Raquel
Urtasun (Karlsruhe Institute of Technology / Toyota Technological Institute
at Chicago) under the
**Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License**
(<https://creativecommons.org/licenses/by-nc-sa/3.0/>).

Under that license, redistribution of this subset is permitted **for
non-commercial research use**, provided that:

1. Attribution is given to the original authors (see citation below).
2. Any derivative work is shared under the same license.
3. The use is non-commercial.

If you intend to use this data commercially, please obtain the data
directly from the official KITTI website (<https://www.cvlibs.net/datasets/kitti>)
and check the licensing terms there.

### Required citation for KITTI

```bibtex
@inproceedings{geiger2012are,
  title     = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  author    = {Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2012}
}
```

## Detector predictions (`data/detector_predictions.txt`)

This file contains predictions from a probabilistic EfficientDet-D0 model
fine-tuned on KITTI for the experiment described in the accompanying paper.
Each line is a Python-literal dictionary with the predicted bounding box,
classification logits, and per-coordinate aleatoric localization
uncertainty.

These predictions are a derivative of the KITTI dataset and are therefore
released under the same **CC BY-NC-SA 3.0** license.

## Uncertainty bins (`data/{low,mid,high}_uncertainty.txt`)

These plain-text files list KITTI image IDs grouped into difficulty bins
based on the per-image average aleatoric localization uncertainty. They
are released under **CC BY-NC-SA 3.0**.

## Re-annotated ground truth (`data/relabeled_ground_truth/`)

The annotation files under `data/relabeled_ground_truth/labels/` are
**produced by the authors of the accompanying paper**. They are corrections
and additions on top of the original KITTI annotations, used as the
gold-standard reference for evaluating annotator performance.

These annotations are a derivative work of KITTI and are therefore
released under **CC BY-NC-SA 3.0**, matching the upstream license.

When using these annotations, please cite both the KITTI paper (above) and
the accompanying paper:

> See `CITATION.cff` and the citation block in `README.md`.

## Source code (`label.py`, `docs/`, scripts)

All source code in this repository is released under the MIT License
(see `LICENSE`). The CC BY-NC-SA 3.0 terms above apply to data only.
