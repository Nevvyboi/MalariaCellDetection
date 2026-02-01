# 🔬 Malaria Cell Detection with CNN

A Convolutional Neural Network for detecting malaria-infected blood cells from microscope images.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Model Architecture](#model-architecture)

---

## Overview

This project implements a CNN to classify blood cell images as either:
- **Parasitized** (infected with malaria)
- **Uninfected** (healthy)

### Why This Matters
Malaria kills over 600,000 people annually. Diagnosis requires trained microscopists to manually examine blood samples - a bottleneck in resource-limited areas. This CNN can assist in automated screening.

---

## Dataset

### Download Instructions

1. **Go to Kaggle:**
   
   https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

2. **Download** the dataset (requires free Kaggle account)

3. **Extract** the zip file

4. **Place** the `cell_images` folder in the `data/` directory:

```
MalariaCellDetection/
├── data/
│   └── cellImages/          ← PUT THE EXTRACTED FOLDER HERE
│       ├── Parasitized/     ← 13,779 infected cell images
│       │   ├── C100P61ThinF_IMG_20150918_144104_cell_162.png
│       │   └── ...
│       └── Uninfected/      ← 13,779 healthy cell images
│           ├── C100P61ThinF_IMG_20150918_144104_cell_128.png
│           └── ...
```

> ⚠️ **Important:** Rename the folder to `cellImages` (or update `config.py` to match your folder name)

### Dataset Details

| Property | Value |
|----------|-------|
| Total Images | 27,558 |
| Infected (Parasitized) | 13,779 (50%) |
| Healthy (Uninfected) | 13,779 (50%) |
| Image Size | ~130×130 pixels (varies) |
| Color | RGB (3 channels) |
| Source | NIH - National Institutes of Health |

---

## Installation

### 1. Clone or Download This Project

```bash
git clone <your-repo-url>
cd MalariaCellDetection
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Follow the [Dataset](#dataset) instructions above.

### 5. Verify Setup

```bash
# Quick test (3 epochs, small subset)
python run.py --quick
```

---

## Project Structure

```
MalariaCellDetection/
│
├── run.py                  # 🚀 Main entry point - RUN THIS
├── README.md               # 📖 This file
├── requirements.txt        # 📦 Python dependencies
├── LICENSE                 # ⚖️ MIT License
│
├── src/                    # 📁 Source code
│   ├── __init__.py        # Package initializer
│   ├── config.py          # ⚙️ All settings and hyperparameters
│   ├── dataset.py         # 📊 Data loading and preprocessing
│   ├── model.py           # 🧠 CNN architecture
│   ├── train.py           # 🏋️ Training loop
│   ├── evaluate.py        # 📈 Evaluation and visualization
│   └── baseline.py        # 📉 Baseline model for comparison
│
├── data/                   # 📂 Dataset folder
│   └── cellImages/        # ← Download and extract here
│       ├── Parasitized/
│       └── Uninfected/
│
├── models/                 # 💾 Saved model weights
│   └── bestModel.pth      # (created during training)
│
└── outputs/                # 📊 Results and visualizations
    ├── training_history.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## Usage

### Quick Test (Verify Everything Works)

```bash
python run.py --quick
```

This runs 3 epochs on a small subset (~2 minutes). Use this to verify your setup works before full training.

### Full Training

```bash
python run.py
```

This runs 35 epochs on the full dataset. Takes approximately:
- **GPU:** 15-30 minutes
- **CPU:** 2-4 hours

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--quick` | Quick test mode (3 epochs, small subset) | `python run.py --quick` |
| `--epochs N` | Custom number of epochs | `python run.py --epochs 50` |
| `--batch-size N` | Custom batch size | `python run.py --batch-size 32` |
| `--learning-rate N` | Custom learning rate | `python run.py --learning-rate 0.0005` |
| `--evaluate-only` | Skip training, evaluate saved model | `python run.py --evaluate-only` |
| `--skip-baseline` | Skip baseline computation | `python run.py --skip-baseline` |

### Examples

```bash
# Full training with default settings
python run.py

# Custom training
python run.py --epochs 50 --batch-size 32

# Quick test
python run.py --quick

# Evaluate existing model without retraining
python run.py --evaluate-only

# Full training but skip baseline (faster)
python run.py --skip-baseline
```

---

## Results

### Expected Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Guessing | ~50% | ~50% |
| Baseline (Logistic Regression) | ~72% | ~70% |
| **CNN** | **~95%+** | **~95%+** |

### Output Files

After training, you'll find these in the `outputs/` folder:

| File | Description |
|------|-------------|
| `training_history.png` | Loss and accuracy curves over epochs |
| `confusion_matrix.png` | Confusion matrix showing predictions vs actual |
| `roc_curve.png` | ROC curve with AUC score |

### Sample Output

```
==============================================================
🔬 MALARIA CELL DETECTION WITH CNN
==============================================================

📁 [1/6] LOADING DATASET
✅ Loaded 27558 images!
🔰 Classes: ['Parasitized', 'Uninfected']
🚂 Train: 19290 | 🧪 Val: 4133 | 🧐 Test: 4135

🧠 [2/6] CREATING MODEL
🏗️  Model: MalariaCellDetection CNN
📊 Trainable parameters: 2,847,234
💻 Device: cuda

🏋️ [3/6] TRAINING MODEL
⏰ Epoch 1 -> Train Loss = 0.4521, Train Acc = 0.7823, Val Acc = 0.8534
...
✅ New best model! Val Accuracy -> 0.9612

📊 [4/6] EVALUATING MODEL
Accuracy ->  0.9587 (95.87%)
Precision -> 0.9623
Recall ->    0.9548
F1 Score ->  0.9585
ROC-AUC ->   0.9891

📉 [5/6] COMPUTING BASELINE
Baseline Accuracy: 0.7234 (72.34%)

📈 MODEL COMPARISON
Metric          Baseline     CNN          Improvement
---------------------------------------------------
Accuracy        0.7234       0.9587       +0.2353
F1 Score        0.7012       0.9585       +0.2573

✅ TRAINING COMPLETE!
```

---

## Model Architecture

```
Input: (batch, 3, 128, 128) - RGB images resized to 128×128

    ┌─────────────────────────────────────────┐
    │ BLOCK 1: Conv2d(3→32) + BatchNorm + ReLU │
    │          MaxPool(2×2)                    │
    │ Output: (batch, 32, 64, 64)              │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │ BLOCK 2: Conv2d(32→64) + BatchNorm + ReLU│
    │          MaxPool(2×2) + Dropout(0.25)    │
    │ Output: (batch, 64, 32, 32)              │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │ BLOCK 3: Conv2d(64→128) + BatchNorm + ReLU│
    │          MaxPool(2×2)                     │
    │ Output: (batch, 128, 16, 16)             │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │ BLOCK 4: Conv2d(128→256) + BatchNorm + ReLU│
    │          MaxPool(2×2) + Dropout(0.25)     │
    │ Output: (batch, 256, 8, 8)                │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │ FLATTEN: (batch, 256×8×8) = (batch, 16384)│
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │ FC1: Linear(16384→512) + ReLU + Dropout(0.5)│
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │ FC2: Linear(512→2)                       │
    │ Output: (batch, 2) - Raw scores          │
    └─────────────────────────────────────────┘
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Size | 128×128 |
| Batch Size | 64 |
| Epochs | 35 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Conv Dropout | 0.25 |
| FC Dropout | 0.5 |

### Data Augmentation

- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.5)
- Random Rotation (±20°)
- Color Jitter (brightness=0.2, contrast=0.2)

---

## Troubleshooting

### "Dataset not found" Error

Make sure your folder structure is:
```
data/
└── cellImages/
    ├── Parasitized/
    └── Uninfected/
```

If your folder is named differently (e.g., `cell_images`), either:
1. Rename it to `cellImages`, OR
2. Update `config.py` line: `self.dataDirectory = os.path.join(root, 'data', 'YOUR_FOLDER_NAME')`

### CUDA Out of Memory

Reduce batch size:
```bash
python run.py --batch-size 32
```

### Training Too Slow

- Make sure you're using GPU: Check that `torch.cuda.is_available()` returns `True`
- Use quick mode for testing: `python run.py --quick`

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## References

- Dataset: [NIH Malaria Cell Images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- Original Paper: Rajaraman et al. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection. PeerJ.
