# Corn-d — Maize Pests & Diseases Detection (Minimal, Reproducible Split)

This repository publishes the **minimal materials** required to verify data hygiene and to reproduce the official splits of the Corn-d dataset:
- fixed split lists,
- class/instance statistics,
- quality checks (duplicate/leakage, inter-annotator agreement),
- and a small helper script **`Evidence.py`**.

> **Split policy:** **split first**, then apply all augmentations **only to the training set** during training. **Validation/Test contain only original images** (no Mosaic or any augmented images).  
> **Random seed:** `2025`.

> **Note:** This repo does **not** host the full images/labels. If you need full training, obtain images from the original sources or a separate release/DOI, then follow the split lists here.

---

## Repository layout

Corn_d/
├─ splits/ # fixed split lists (frozen with seed=2025)
│ ├─ train.txt
│ ├─ val.txt
│ ├─ test.txt
│ ├─ train_stems.txt
│ ├─ val_stems.txt
│ └─ test_stems.txt
├─ stats/ # distributions and counts
│ ├─ split_distribution.csv
│ ├─ counts_after_step1.csv
│ └─ counts_after_step2.csv
├─ quality/ # data hygiene & agreement materials
│ ├─ duplicates/
│ │ ├─ dup_report_sha1.txt # exact duplicate scan (SHA-1)
│ │ └─ dup_report_ahash.txt # perceptual duplicate scan (aHash)
│ ├─ leakage_checks/
│ │ ├─ leakage_check.txt # cross-split zero-intersection checks
│ │ └─ invalid_cid_report.txt
│ └─ iaa/
│ ├─ class_names.txt
│ ├─ iaa_list.csv
│ ├─ selected_stems.txt
│ └─ teacher_adjust_log.csv
├─ Evidence.py # helper to print a concise evidence summary
└─ README.md
---

## Classes

The class list (index → name) is aligned with `quality/iaa/class_names.txt`:

---

## Fixed splits & hygiene

- **Seeded split:** All lists in `splits/` were generated with `seed=2025` and are frozen.  
- **No leakage:** Train/Val/Test have **zero overlap**. See:
  - `quality/leakage_checks/leakage_check.txt`
  - `quality/duplicates/dup_report_sha1.txt` and `dup_report_ahash.txt` (contain thresholds and matches, if any).
- **Stats:** Class and instance distributions are provided in `stats/`.

**Quick self-checks (recommended):**
1) The line counts of `train.txt / val.txt / test.txt` should match the number of images you place in the corresponding split.  
2) Every image has a label file with the same stem and a valid class ID from `class_names.txt`.  
3) Re-run your own duplicate scan before training (especially if you re-collect images).

---

## Using `Evidence.py`

`Evidence.py` prints a concise **evidence summary** based on the materials in `splits/`, `stats/`, and `quality/` (e.g., split sizes, cross-split intersections, duplicate scan notes, and IAA subset size).

```bash
python Evidence.py
