# ⚡ MICRONAS ENGINE (Hackathon Edition)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-16A34A.svg)](https://xgboost.ai/)
[![PySide6](https://img.shields.io/badge/PySide6-GUI-41CD52.svg)](https://wiki.qt.io/Qt_for_Python)

**MICRONAS ENGINE** is a fully autonomous, dataset-aware, and highly generalizable AutoML platform. It accepts any dataset (CSV, Images, Sequences, Torchvision) alongside a natural language prompt, and automatically engineers, searches, trains, and exports the perfect machine learning architecture tailored to your data's specific shape and complexity.

Designed for maximum stability and generalization, MICRONAS actively monitors the gap between training and validation accuracy to intelligently expand depth or aggressively punish overfitting via dynamic dropout scaling and pruning.

---

## 🎯 Core Features

- **🧠 Dataset-Adaptive Multi-Model Selection**: The system natively evaluates and cross-compares entirely different model families:
  - *Tabular*: MLPs, Random Forests, XGBoost.
  - *Images*: Small, Medium, and Deep Convolutional Blocks.
  - *Text/Time-Series*: LSTMs, GRUs, Temporal 1D-CNNs.
- **🛡️ Overfitting & Underfitting Control System**:
  - If `train_acc >> val_acc` (> 10%), the Engine actively *increases dropout* and *prunes network layers*.
  - If both accuracies are low, the Engine explicitly mutates the architecture to be deeper/wider and increments training epochs.
- **📈 Validation-First Proxy Search**: Every candidate architecture is subjected to an ultra-fast, genuine 2-epoch training run. Fitness is calculated purely by actual **validation accuracy** (not training accuracy or zero-cost guesses), ensuring true generalization.
- **🚀 Advanced Training Pipeline**: Uses `AdamW` (weight decay `1e-4`), `ReduceLROnPlateau` scheduling, Gradient Clipping, Image Augmentation (`RandomHorizontalFlip`, `RandomRotation`), and strict 80/20 deterministic data splits.
- **🧬 Ensemble Generator**: If two top-performing neural architectures are within 2% validation accuracy of one another at the end of the search, the Engine dynamically packages them into an `EnsembleWrapper` to combine predictions.
- **📊 Real-Time Visualization & Explainability**: A stunning PySide6 GUI provides live updates of the Training curves, an interactive "Cross-Model Comparison" bar chart (Train vs Val Acc for top candidates), and an explicitly generated `EXPLAINABILITY.md` report detailing *why* the architecture won.

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/micronas-engine.git
   cd micronas-engine
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r micronas/requirements.txt
   ```

---

## 🚀 How to Run

Launch the MICRONAS Engine GUI:

```bash
python micronas/main.py
```
*(Ensure `micronas` is in your `PYTHONPATH` if running from outside the directory: `PYTHONPATH=. python micronas/main.py`)*

---

## 🖥️ Usage Guide

### 1. Loading a Dataset
Use the **Dataset** dropdown in the UI. You can select:
- Pre-built datasets (`MNIST`, `CIFAR10`, `mock.csv`).
- **Choose File/Folder...**: Opens a native file dialog.
  - Select a `.csv` file for Tabular / Time-Series data (assumes the last column is the target variable).
  - Cancel the file dialog to open a Folder dialog, and select a directory containing subfolders of images (standard ImageFolder format).

### 2. Natural Language Prompts
Type your goal into the prompt box. The engine parses keywords to adjust the AutoML constraints:
- *"Train a fast classifier"* -> Increases the latency/size penalty.
- *"Build a highly accurate model / Minimize errors"* -> Unlocks deeper parameter limits, increases base training epochs (3x multiplier), and boosts base dropout for generalization.

### 3. The Live Pipeline
Click **🚀 START MICRONAS ENGINE**. You will see:
1. **AI Thinking Log**: Real-time logs of the specific architectures (e.g., `Wide MLP`, `XGBoost`, `Deep CNN`) being built and evaluated.
2. **Training Loss Chart**: Real-time updates as the current proxy model trains.
3. **Cross-Model Comparison Graph**: At the end of the search, a bar chart maps the Train vs Val Acc of the top 5 candidates.
4. **Live Predictor**: Once the final, fully-retrained model is ready, a "Test Final Model" button unlocks allowing you to upload a new CSV/Image and instantly receive a prediction alongside a **Softmax Confidence %**.

---

## 📦 Output Structure

Once the engine finishes, it seamlessly outputs everything you need to deploy into `project_output/`.

```text
project_output/
├── model.pt (or .pkl)           # The final trained weights (PyTorch or Joblib for Trees)
├── architecture_memory.json     # The NAS engine's memory of the best config
├── predict.py                   # A standalone script to run inference on new data
├── train.py                     # A stub script demonstrating how to resume training
├── requirements.txt             # Bare-minimum dependencies for deployment
├── confusion_matrix.png         # Heatmap of the final validation errors
├── results.json                 # Dictionary containing the final val_acc and train_loss
├── README.md                    # Auto-generated instructions for the exported model
└── EXPLAINABILITY.md            # A detailed Markdown report explaining WHY this architecture won
```

---

## ⚠️ Failsafe Reliability (Hackathon Guarantee)
MICRONAS is designed around the core principle: **STABILITY > ACCURACY > COMPLEXITY**.
If an architecture mutates into an invalid shape (e.g. collapsing spatial dimensions) or runs out of GPU memory during the proxy search, the system catches the internal PyTorch crash, explicitly marks the candidate's fitness as `-1`, and cleanly moves to the next architecture. The UI will **never crash**.

---

*Built for generalization, speed, and reliability.*
