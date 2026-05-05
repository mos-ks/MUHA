---
title: Bounding Box Labeling Tool
emoji: 📦
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.27.1"
app_file: label.py
pinned: false
license: mit
---

# MUHA &nbsp;·&nbsp; Model Uncertainty &rarr; Human Attention

Code, data, and project page for the paper:

> **From Model Uncertainty to Human Attention: Localization-Aware Visual Cues for Scalable Annotation Review**
> Moussa Kassem Sbeyti<sup>†</sup>, Joshua Holstein<sup>†</sup>, Philipp Spitzer, Nadja Klein, Gerhard Satzger.
> *Under review (2026).*
>
> <sup>†</sup>Equal contribution. Authors are permitted to list their name first.
>
> Project page: <https://mos-ks.github.io/MUHA/>

## Citation

A formal citation will be added once the paper is accepted. Until then,
please cite using the metadata in [`CITATION.cff`](CITATION.cff) or the
BibTeX block below:

```bibtex
@unpublished{KaSbHoSpNaSa2026,
  title  = {From Model Uncertainty to Human Attention:
            Localization-Aware Visual Cues for Scalable Annotation Review},
  author = {Kassem Sbeyti, Moussa and Holstein, Joshua and Spitzer, Philipp
            and Klein, Nadja and Satzger, Gerhard},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## What's in this repository

```
.
├── label.py                       # Streamlit labeling tool used in the study
├── build_site_data.py             # Builds docs/data/annotations.json
├── group_images_by_uncertainty.py # Re-derives the difficulty bins
├── pass_hasher.py                 # Helper to hash auth passwords
├── config.yaml                    # Auth config for the Streamlit app
├── data/
│   ├── ground_truth/              # 97 KITTI experimental images and the
│   │   ├── images/                # original KITTI annotations for them
│   │   └── labels/
│   ├── relabeled_ground_truth/    # Authors' corrected annotations (gold)
│   │   ├── labels/                # Per-image KITTI-format labels (955 boxes)
│   ├── relabeled_ground_truth.txt # Predictions paired with relabeled GT
│   ├── detector_predictions.txt   # Probabilistic-EfficientDet predictions
│   ├── low_uncertainty.txt        # Difficulty bins (image IDs, one per line)
│   ├── mid_uncertainty.txt
│   └── high_uncertainty.txt
├── docs/                          # GitHub Pages project site
├── analysis/                      # R code, input data, and output figures
│   ├── Analysis.R                 # Full statistical analysis (Sections 1-9)
│   ├── requirements.R             # Installs required R packages
│   ├── 00_data/                   # Trial-level and box-level study data
│   └── 01_figures/                # Generated manuscript figures (PDF)
└── LICENSE, LICENSE-DATA.md, CITATION.cff
```

### About the 97 images

The repository ships exactly the **97 images** that were shown to
participants during the experiment and re-annotated by the authors.
Re-annotation enlarged the original 566 KITTI bounding boxes into the
**955-box gold standard** (Table S4 of the paper); the per-image KITTI-format
files in `data/relabeled_ground_truth/labels/` are the authors' gold
standard, and `data/relabeled_ground_truth.txt` is the pred-paired form
that also carries the model-side metadata (logits, aleatoric uncertainty,
etc.).

## Data licensing — please read

The KITTI images and original annotations under `data/ground_truth/` are
**not the authors' work**. They are redistributed here under the KITTI
**CC BY-NC-SA 3.0** license, which permits non-commercial research use
with attribution and share-alike. See
[`LICENSE-DATA.md`](LICENSE-DATA.md) for the full attribution and the
required citation. For commercial use, obtain the data directly from
<https://www.cvlibs.net/datasets/kitti>.

The labeling tool source code is released under the MIT License
([`LICENSE`](LICENSE)). The detector predictions and the
authors' re-annotated ground truth are released under CC BY-NC-SA 3.0 to
remain compatible with the upstream dataset.

## Running the labeling tool

```bash
pip install -r requirements.txt
streamlit run label.py
```

The app expects a query string for authentication and treatment assignment:

```
http://localhost:8501/?username=USER&password=PASS&PROLIFIC_PID=ID&treatment=TREATMENT
```

`TREATMENT` is one of:

| value      | what it does                                                              |
|------------|---------------------------------------------------------------------------|
| `sigma_0`  | Baseline interface (control condition).                                   |
| `sigma_1`  | Uncertainty-aware interface (treatment condition reported in the paper).  |
| `sigma_2`  | Uncertainty + low-uncertainty boxes dimmed (development variant).         |

Append `&labeler=relabel` to enter the authors' relabeling pass. In that
mode the per-participant 5-mid + 10-high uncertainty-bin sample is
bypassed and the full set of images on disk is shown, which is how the
authors produced the gold-standard annotations reported in Table S4.

## Project page

A static project page lives under [`docs/`](docs/) and is auto-deployed via
GitHub Pages. It includes an interactive viewer that lets readers step
through the 136 candidate images and compare:

1. **Original KITTI ground truth** (red dashed)
2. **Detector predictions** (blue, colour-coded by uncertainty in the
   uncertainty view)
3. **Authors' re-annotated ground truth** (green)

Use the **Prev / Next** buttons or the keyboard `←` / `→` arrows to navigate.

## Reproducing the analysis

The randomized controlled study was conducted on Prolific with N = 120
participants and 1,800 annotation trials. Statistical analyses were carried
out in **R 4.4.1** using `tidyverse`, `lme4`, `lmerTest`, and `coin`. See
the *Statistical Analysis* section of the paper for the complete
specification.

All analysis code and data live under [`analysis/`](analysis/):

```bash
Rscript analysis/requirements.R   # Install R package dependencies
Rscript analysis/Analysis.R       # Run from the repository root
```

`Analysis.R` reads the trial- and box-level data from `analysis/00_data/`,
prints every statistical result reported in the paper to the console, and
writes the manuscript figures (Figures 1–3) to `analysis/01_figures/`.

## Contact

For questions about the paper, please contact the corresponding authors
(see paper). For issues with this repository, please open a GitHub issue.
