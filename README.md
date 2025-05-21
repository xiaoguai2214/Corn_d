# 🌽 Corn-d: A Field-Collected Dataset for Maize Pest and Disease Detection

![License: CC BY 4.0](https://img.shields.io/badge/license-CC--BY--4.0-brightgreen.svg)

This repository hosts the **Corn-d dataset**, a publicly available image dataset for evaluating pest and disease detection models in real-world maize fields. It was constructed to support robust object detection research under complex environmental conditions such as dense targets, background clutter, occlusion, and multi-source interference.

---

## 📂 Dataset Structure

The dataset is organized in the following format (YOLO format by default):

```
Corn-d/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── classes.txt
```

- `images/`: RGB images captured from real maize fields.
- `labels/`: Text files in YOLO format: `[class_id x_center y_center width height]`
- `classes.txt`: Lists the class names, one per line.

---

## 🐛 Categories

| Class ID | Class Name        |
|----------|-------------------|
| 0        | Healthy           |
| 1        | Common_Rust       |
| 2        | Blight            |
| 3        | Gray_Leaf_Spot    |

---

## 📊 Statistics

- **Total Images**: ~7,300
- **Annotation Format**: YOLOv11-compatible
- **Image Resolution**: Variable (e.g., 640×480, 1280×720)
**Data Sources**:
  - Based on IP02 (public pest dataset)
  - Combined with samples from PlantVillage and PlantDoc (plant disease datasets)
- **Data Processing**:
  - All images were re-labeled using a unified class scheme;
  - Data augmentation methods applied: flipping, rotation, blur, contrast enhancement, etc.
  - Final annotations are structured in YOLO format across train/val/test splits.

---

## 📈 Suggested Use

The dataset was used in the paper:

> Yuan, Lijun & Wang, Fang. *MTA-YOLOv11: Detection of Maize Pests and Diseases under Complex Scenes*. [Preprint Link (if available)]

This dataset can be used with standard YOLO implementations to reproduce detection results on complex agricultural scenes.

---

## 🚀 How to Use

```bash
git clone https://github.com/yourusername/Corn-d.git
cd Corn-d
```

Then link the dataset to your YOLO training code:

```bash
python train.py --data corn.yaml --weights yolov11n.pt --epochs 100
```

Example `corn.yaml`:

```yaml
train: ./Corn-d/images/train
val: ./Corn-d/images/val

nc: 7
names: ['Healthy', 'Common_Rust', 'Blight', 'Gray_Leaf_Spot', 'Corn_borer', 'Army_worm', 'Aphids']
```

---

## 📄 License

This dataset is released under the [Creative Commons Attribution 4.0 License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  
You are free to share and adapt the material, as long as proper attribution is given.

---

## 📬 Contact

If you use this dataset in your research, please cite the original paper.  
For questions or collaborations, contact the corresponding author:

- **Lijun Yuan**  
- Email: `2214041643@qq.com`
