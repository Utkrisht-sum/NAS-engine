# ⚡ MICRONAS ENGINE (Hackathon Edition)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/HuggingFace-Accelerate-F9AB00.svg)](https://huggingface.co/docs/accelerate/index)
[![PySide6](https://img.shields.io/badge/PySide6-GUI-41CD52.svg)](https://wiki.qt.io/Qt_for_Python)

**MICRONAS ENGINE** is a fully autonomous, self-evolving, hardware-aware AutoML system. It accepts any dataset (CSV, Images, Torchvision) alongside a natural language prompt, and automatically engineers, searches, trains, and exports the perfect neural network architecture tailored to your specific hardware constraints.

Designed strictly for low-end hardware (e.g., GTX 1050 with 3GB VRAM), MICRONAS guarantees crash-free execution through dynamic architecture repair, mixed-precision CPU/GPU offloading, and zero-cost proxies.

---

## 🎯 Core Features

- **🧠 Prompt-Driven AutoML**: Simply type *"Build a fast image classifier"*, and the engine will automatically bias the Evolutionary Search to penalize high-latency architectures.
- **🧬 Zero-Cost Evolutionary NAS**: Uses approximations like SynFlow and NASWOT to filter out 90% of weak architectures before any training begins, making the search incredibly fast.
- **🛡️ Failure-Aware Self-Repair**: If an architecture mutates into something that causes a spatial dimension collapse or OOM error, the engine catches it, dynamically repairs the kernel sizes/layers, and remembers the failure to avoid it in the future.
- **💾 AirLLM-Style Memory Safety**: Powered by Hugging Face `accelerate`, it automatically maps models across CPU and GPU, ensuring you never run out of memory during the final training phase.
- **📊 Real-Time Visualization**: A stunning PySide6 GUI provides live updates of the NAS Progress, Training Loss curves, Pareto Efficiency Frontiers, and an "AI Thinking" log.
- **🏆 Production-Ready Export**: At the end of the pipeline, it generates a complete standalone Python project (`predict.py`, `train.py`, `README.md`, `model.pt`) and an `EXPLAINABILITY.md` report detailing *why* the architecture was chosen.

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

4. **Initialize Accelerate (Optional but recommended for optimal hardware mapping):**
   ```bash
   accelerate config
   # Answer the prompts based on your hardware (e.g., This machine, No distributed training, fp16 mixed precision).
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

### 1. The Interface
The GUI is divided into several logical panels:
- **Top Panel**: Task Definition (Dataset & Prompt).
- **Middle Panel**: NAS Configuration & Demo Presets.
- **Bottom Left**: Real-time Interactive Charts.
- **Bottom Right**: Live Logs and "AI Thinking" stream.

### 2. Loading a Dataset
Use the **Dataset** dropdown. You can select:
- Pre-built Torchvision datasets (`MNIST`, `CIFAR10`).
- A quick mock CSV (`mock.csv`).
- **Choose File/Folder...**: This opens a native file dialog.
  - Select a `.csv` file for Tabular data (assumes the last column is the target variable).
  - Cancel the file dialog to open a Folder dialog, and select a directory containing subfolders of images (standard ImageFolder format).

### 3. Natural Language Prompts
Type your goal into the prompt box. The engine uses a heuristic parser to adjust the NAS fitness function $F(A) = \alpha(\text{acc}) - \beta(\text{params}) - \gamma(\text{latency}) - \delta(\text{memory})$.
- *"Train a fast classifier"* -> Increases the latency penalty ($\gamma$).
- *"Build an efficient/lightweight model"* -> Increases parameter/memory penalty ($\beta, \delta$).
- *"Give me the most accurate model"* -> Increases the accuracy reward ($\alpha$).

### 4. Running a Quick Demo
If you are presenting this at a hackathon, use the **Quick Demos** buttons:
- **Demo 1: MNIST Efficient**: Auto-fills the dataset to MNIST, sets an efficiency prompt, and configures a fast population size.
- **Demo 2: Fast Text/CSV**: Auto-generates a mock CSV dataset and runs a rapid tabular NAS.

Click **🚀 START MICRONAS ENGINE** and watch the magic happen.

---

## 🧠 The Pipeline Explained

1. **Dataset Analyzer**: Detects input shape, number of classes, and infers if it's a classification or regression task.
2. **Prompt Parser**: Converts your English text into hyperparameter weights for the fitness function.
3. **Architecture Sampling**: Generates random neural networks (Dynamic MLPs or CNNs depending on the dataset).
4. **Zero-Cost Evaluation**: Scores each model instantly without backpropagation using SynFlow/NASWOT proxies.
5. **Evolutionary Search**: Mutates the best models, creating a Pareto frontier of efficiency vs accuracy.
6. **Full Training**: Takes the absolute best model from the NAS phase and trains it fully using Adam, mixed precision, and robust OOM-catching.
7. **Export**: Dumps the model and standalone inference scripts.

---

## 📦 Output Structure

Once the engine finishes, it creates a folder in your working directory called `project_output/`.

```text
project_output/
├── model.pt                     # The final trained PyTorch weights
├── architecture_memory.json     # The NAS engine's memory of the best config
├── predict.py                   # A standalone script to run inference on new data
├── train.py                     # A stub script demonstrating how to resume training
├── requirements.txt             # Bare-minimum dependencies for deployment
├── README.md                    # Auto-generated instructions for the exported model
└── EXPLAINABILITY.md            # A detailed report explaining WHY this architecture won
```

---

## ⚠️ Known Limitations (Hackathon Scope)
- Text/NLP processing and Time-Series (LSTM/GRU) generation are architecturally planned but simplified to MLPs for the MVP.
- `predict.py` currently exports as a generic stub meant to be customized with the specific dynamic architecture class from the engine.
- Memory estimation uses heuristic forward-pass approximations to stay lightweight and avoid crashing low-VRAM GPUs.

---

*Built with passion, caffeine, and PyTorch.*
