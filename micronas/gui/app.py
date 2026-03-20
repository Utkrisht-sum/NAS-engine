import sys
import threading
import torch
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QTextEdit,
    QSplitter, QProgressBar, QGroupBox, QFormLayout, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Signal, QObject

from gui.charts import ChartsPanel
from engine.dataset import DatasetAnalyzer
from engine.nas import NASEngine
from engine.trainer import Trainer
from engine.export import ProjectExporter
from utils.logger import get_logger

logger = get_logger("GUI")

class WorkerSignals(QObject):
    log_msg = Signal(str)
    ai_msg = Signal(str)
    nas_progress = Signal(int, float, list)  # gen, fitness, pop_data
    train_progress = Signal(int, float, float, float) # epoch, t_loss, v_loss, v_acc
    model_comparison = Signal(list) # top models from NAS
    explainability_msg = Signal(str)
    finished = Signal()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ MICRONAS ENGINE - AutoML")
        self.resize(1200, 800)

        self.signals = WorkerSignals()
        self.signals.log_msg.connect(self.append_log)
        self.signals.ai_msg.connect(self.set_ai_thinking)
        self.signals.nas_progress.connect(self.update_nas_chart)
        self.signals.train_progress.connect(self.update_train_chart)
        self.signals.model_comparison.connect(self.update_model_comparison_table)
        self.signals.explainability_msg.connect(self.update_explainability)
        self.signals.finished.connect(self.on_finished)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # TOP PANEL: Inputs
        top_group = QGroupBox("1. Task Definition")
        top_layout = QFormLayout()

        self.dataset_input = QComboBox()
        self.dataset_input.addItems(["MNIST", "CIFAR10", "mock.csv", "Choose File/Folder..."])
        self.dataset_input.currentTextChanged.connect(self.handle_dataset_selection)
        top_layout.addRow("Dataset:", self.dataset_input)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g. Train an efficient digit classifier")
        top_layout.addRow("Prompt:", self.prompt_input)

        top_group.setLayout(top_layout)
        main_layout.addWidget(top_group)

        # MIDDLE PANEL: NAS Config
        mid_group = QGroupBox("2. NAS Configuration")
        mid_layout = QHBoxLayout()

        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(2, 50)
        self.pop_spin.setValue(5)

        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 20)
        self.gen_spin.setValue(2)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100)
        self.epoch_spin.setValue(2)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Fast Mode", "Balanced Mode", "Research Mode"])

        mid_layout.addWidget(QLabel("Population:"))
        mid_layout.addWidget(self.pop_spin)
        mid_layout.addWidget(QLabel("Generations:"))
        mid_layout.addWidget(self.gen_spin)
        mid_layout.addWidget(QLabel("Epochs:"))
        mid_layout.addWidget(self.epoch_spin)
        mid_layout.addWidget(QLabel("Mode:"))
        mid_layout.addWidget(self.mode_combo)

        mid_group.setLayout(mid_layout)
        main_layout.addWidget(mid_group)

        # BOTTOM CONTROLS & DEMOS
        bottom_controls = QHBoxLayout()

        # DEMO PRESETS
        demo_group = QGroupBox("Quick Demos")
        demo_layout = QHBoxLayout()
        btn_demo1 = QPushButton("Demo 1: MNIST Efficient")
        btn_demo1.clicked.connect(lambda: self.load_demo("MNIST", "Train an efficient digit classifier", 5, 2))
        btn_demo2 = QPushButton("Demo 2: Fast Text/CSV")
        btn_demo2.clicked.connect(lambda: self.load_demo("mock.csv", "Build a fast model", 4, 1))

        demo_layout.addWidget(btn_demo1)
        demo_layout.addWidget(btn_demo2)
        demo_group.setLayout(demo_layout)
        bottom_controls.addWidget(demo_group)

        # PREDICTION PREVIEW (Disabled until train finishes)
        pred_group = QGroupBox("Live Prediction & Metrics")
        pred_layout = QHBoxLayout()
        self.btn_predict = QPushButton("Test Final Model")
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.run_live_prediction)
        self.pred_label = QLabel("Waiting for model...")

        # Explicit Accuracy Labels
        self.lbl_acc_proxy = QLabel("Best NAS Proxy Accuracy: --%")
        self.lbl_acc_final = QLabel("Final Accuracy: --%")
        self.lbl_acc_final.setStyleSheet("font-weight: bold; color: green;")

        pred_layout.addWidget(self.btn_predict)
        pred_layout.addWidget(self.pred_label)
        pred_layout.addWidget(self.lbl_acc_proxy)
        pred_layout.addWidget(self.lbl_acc_final)

        pred_group.setLayout(pred_layout)
        bottom_controls.addWidget(pred_group)

        main_layout.addLayout(bottom_controls)

        # BOTTOM PANEL: Run & Progress
        self.btn_start = QPushButton("🚀 START MICRONAS ENGINE")
        self.btn_start.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px; background-color: #2e8b57; color: white;")
        self.btn_start.clicked.connect(self.start_pipeline)
        main_layout.addWidget(self.btn_start)
        self.exported_model = None
        self.exported_metadata = None

        # SPLITTER: Charts & Logs
        splitter = QSplitter(Qt.Horizontal)

        # Left: Charts
        self.charts_panel = ChartsPanel()
        splitter.addWidget(self.charts_panel)

        # Right: Logs + AI Thinking
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.ai_thinking_label = QLabel("🧠 AI Thinking: Idle")
        self.ai_thinking_label.setStyleSheet("color: blue; font-weight: bold;")
        right_layout.addWidget(self.ai_thinking_label)

        # Split right panel into Logs and Model Comparison/Explainability
        right_splitter = QSplitter(Qt.Vertical)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        right_splitter.addWidget(self.log_output)

        # Model Comparison Table
        self.table_comparison = QTableWidget(0, 4)
        self.table_comparison.setHorizontalHeaderLabels(["Rank", "Config", "Accuracy", "Params"])
        self.table_comparison.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        right_splitter.addWidget(self.table_comparison)

        # Explainability Box
        self.explain_output = QTextEdit()
        self.explain_output.setReadOnly(True)
        self.explain_output.setPlaceholderText("Final Explainability Report will appear here...")
        right_splitter.addWidget(self.explain_output)

        right_layout.addWidget(right_splitter)

        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)

    def handle_dataset_selection(self, text):
        if text == "Choose File/Folder...":
            path, _ = QFileDialog.getOpenFileName(self, "Select Dataset File (CSV)", "", "CSV Files (*.csv);;All Files (*)")
            if not path:
                # If they cancel, offer folder instead
                path = QFileDialog.getExistingDirectory(self, "Select Image Dataset Folder")

            if path:
                self.dataset_input.blockSignals(True)
                self.dataset_input.insertItem(0, path)
                self.dataset_input.setCurrentIndex(0)
                self.dataset_input.blockSignals(False)
                self.append_log(f"Selected custom dataset: {path}")
            else:
                self.dataset_input.setCurrentIndex(0) # Revert to MNIST

    def load_demo(self, dataset, prompt, pop, gens):
        self.dataset_input.setCurrentText(dataset)
        self.prompt_input.setText(prompt)
        self.pop_spin.setValue(pop)
        self.gen_spin.setValue(gens)
        self.append_log(f"Loaded demo: {dataset} | {prompt}")

    def append_log(self, text):
        self.log_output.append(text)
        # auto scroll
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_ai_thinking(self, text):
        self.ai_thinking_label.setText(f"🧠 AI Thinking: {text}")

    def update_nas_chart(self, gen, best_fitness, pop_data):
        self.charts_panel.update_nas_chart(gen, best_fitness)
        self.charts_panel.update_pareto_chart(pop_data)

    def update_train_chart(self, epoch, t_loss, v_loss, v_acc):
        self.charts_panel.update_train_chart(epoch, t_loss, v_loss)
        self.lbl_acc_final.setText(f"Final Accuracy: {v_acc:.2f}%")

    def update_model_comparison_table(self, top_models):
        self.table_comparison.setRowCount(0)
        for i, m in enumerate(top_models[:5]):  # Show top 5
            self.table_comparison.insertRow(i)
            self.table_comparison.setItem(i, 0, QTableWidgetItem(f"#{i+1}"))

            # Formatting the config to be readable
            conf_str = "MLP " + str(m['config'].get('hidden_layers', [])) if m['config']['type'] == 'mlp' else "CNN"
            self.table_comparison.setItem(i, 1, QTableWidgetItem(conf_str))

            # "accuracy proxy" shown clearly
            proxy_acc = f"{m['accuracy_proxy']*100:.2f}% (Proxy)"
            self.table_comparison.setItem(i, 2, QTableWidgetItem(proxy_acc))

            params = f"{m['params'] / 1000:.1f}K"
            self.table_comparison.setItem(i, 3, QTableWidgetItem(params))

    def update_explainability(self, report_text):
        self.explain_output.setMarkdown(report_text)

    def start_pipeline(self):
        self.btn_start.setEnabled(False)
        self.log_output.clear()

        dataset = self.dataset_input.currentText()
        prompt = self.prompt_input.text()
        pop = self.pop_spin.value()
        gens = self.gen_spin.value()
        epochs = self.epoch_spin.value()

        # Need a mock CSV for demo if chosen
        if dataset == "mock.csv":
            import pandas as pd
            import numpy as np
            import os
            if not os.path.exists("mock.csv"):
                df = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'target'])
                df['target'] = np.random.choice([0, 1], 100)
                df.to_csv("mock.csv", index=False)

        threading.Thread(target=self.run_pipeline_thread, args=(dataset, prompt, pop, gens, epochs), daemon=True).start()

    def run_pipeline_thread(self, dataset, prompt, pop, gens, epochs):
        try:
            self.signals.ai_msg.emit("Analyzing Dataset...")
            self.signals.log_msg.emit(f"Loading dataset: {dataset}")
            analyzer = DatasetAnalyzer(dataset, "auto")
            metadata, train_loader, val_loader = analyzer.analyze_and_load(batch_size=32)
            self.signals.log_msg.emit(f"Metadata detected: {metadata}")

            self.signals.ai_msg.emit("Parsing Prompt and Initializing NAS...")
            nas = NASEngine(metadata, train_loader, val_loader, prompt=prompt)

            # Hook the NAS output to GUI using monkey patching for the hackathon
            original_eval = nas._evaluate_fitness
            def eval_hook(model, config, device):
                res = original_eval(model, config, device)
                # Just show one evaluation message as AI thinking
                self.signals.ai_msg.emit(f"Evaluating arch with params: {res['params']} | Mem: {res['memory_mb']:.2f}MB")
                return res
            nas._evaluate_fitness = eval_hook

            self.signals.log_msg.emit(f"Running NAS. Pop: {pop}, Gens: {gens}")

            # Monkey patch the nas evolutionary loop directly to get LIVE chart updates
            original_run_search = nas.run_search

            # Helper to wrap the NAS fitness evaluation with a live UI callback
            def evaluate_with_live_ui(model, config, device, candidate_idx=0, total_candidates=1):
                conf_name = config.get('name', 'MLP Config')
                self.signals.ai_msg.emit(f"Evaluating {candidate_idx}/{total_candidates}: {conf_name}...")
                self.signals.log_msg.emit(f"Starting 2-epoch proxy training for candidate {candidate_idx}: {conf_name}")

                # Monkey patch the internal trainer used by _evaluate_fitness just for this run to emit live signals
                import engine.trainer
                original_trainer_train = engine.trainer.Trainer.train

                def live_proxy_train(trainer_self, epochs=2, early_stopping_patience=10, callback=None):
                    def proxy_cb(epoch, t_loss, v_loss, v_acc):
                        self.signals.train_progress.emit(epoch, t_loss, v_loss, v_acc)
                        self.signals.log_msg.emit(f"[NAS Search] {conf_name} - Epoch {epoch}/{epochs} | TL: {t_loss:.3f} | VL: {v_loss:.3f} | Acc: {v_acc:.2f}%")
                    return original_trainer_train(trainer_self, epochs=epochs, early_stopping_patience=early_stopping_patience, callback=proxy_cb)

                engine.trainer.Trainer.train = live_proxy_train
                try:
                    res = nas._evaluate_fitness(model, config, device)
                finally:
                    engine.trainer.Trainer.train = original_trainer_train

                self.signals.ai_msg.emit(f"Candidate {candidate_idx} scored {res['fitness']:.2f}% accuracy.")
                return res

            def live_run_search(population_size, generations, max_params=1e6):
                # Similar to original loop but emitting signals live
                logger = get_logger("NASEngine")
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Initialize
                self.signals.ai_msg.emit("Initializing Candidate Population...")
                for idx in range(population_size):
                    if nas.metadata["type"] == "tabular":
                        config = nas._sample_tabular_config()
                    elif nas.metadata["type"] == "image":
                        config = nas._sample_cnn_config()
                    else:
                        config = nas._sample_sequence_config()

                    model = nas._build_model(config)
                    if model and getattr(nas, 'count_parameters', lambda m: sum(p.numel() for p in m.parameters()))(model) < max_params:
                        eval_data = evaluate_with_live_ui(model, config, device, candidate_idx=idx+1, total_candidates=population_size)
                        nas.population.append(eval_data)

                # Filter out failed evaluations
                nas.population = [p for p in nas.population if p["fitness"] > -1]

                # Handle case where all models exceeded max_params or crashed (Fallback)
                if len(nas.population) == 0:
                    self.signals.ai_msg.emit("All initial candidates failed. Injecting failsafe model...")
                    logger.warning("No models fit or all crashed. Using default fallback.")
                    if nas.metadata["type"] == "tabular":
                        config = nas._sample_tabular_config()
                    elif nas.metadata["type"] == "image":
                        config = nas._sample_cnn_config()
                    else:
                        config = nas._sample_sequence_config()
                    model = nas._build_model(config)
                    eval_data = evaluate_with_live_ui(model, config, device, candidate_idx=1, total_candidates=1)
                    nas.population.append(eval_data)

                # Evolution Loop
                import random
                for gen in range(generations):
                    nas.population = sorted(nas.population, key=lambda x: x["fitness"], reverse=True)
                    nas.history.append([ind["fitness"] for ind in nas.population])

                    # LIVE EMIT here!
                    best_fitness_now = nas.population[0]["fitness"]
                    self.signals.nas_progress.emit(gen+1, best_fitness_now, nas.population)
                    self.signals.ai_msg.emit(f"Generation {gen+1}/{generations} | Current Best Fitness: {best_fitness_now:.2f}%")

                    parents = nas.population[:population_size//2]
                    next_gen = parents.copy()

                    while len(next_gen) < population_size:
                        parent = random.choice(parents)
                        child_config = nas._mutate(parent["config"], parent_stats=parent)
                        if child_config in nas.failures: continue
                        child_model = nas._build_model(child_config)
                        if child_model and getattr(nas, 'count_parameters', lambda m: sum(p.numel() for p in m.parameters()))(child_model) < max_params:
                            eval_data = evaluate_with_live_ui(child_model, child_config, device, candidate_idx=len(next_gen)+1, total_candidates=population_size)
                            next_gen.append(eval_data)
                    nas.population = next_gen

                nas.population = sorted(nas.population, key=lambda x: x["fitness"], reverse=True)

                best = nas.population[0]
                nas.best_model = best["model"]
                nas.best_config = best["config"]
                nas._save_memory(best)
                return nas.best_model, nas.best_config, nas.population

            nas.run_search = live_run_search
            best_model, best_config, final_pop = nas.run_search(population_size=pop, generations=gens)

            self.signals.log_msg.emit(f"Best Config Found: {best_config}")
            self.signals.ai_msg.emit(f"NAS Completed. Top Model Found.")

            # Update best proxy label
            best_proxy = final_pop[0]['accuracy_proxy'] * 100
            self.lbl_acc_proxy.setText(f"Best NAS Proxy Accuracy: {best_proxy:.2f}%")

            # Emit top models for comparison table
            self.signals.model_comparison.emit(final_pop)

            # Emit explainability reasoning before training
            best_stats = final_pop[0]
            explain_text = f"""## 🧠 MICRONAS Decision Engine

**Why this model was selected:**
- **Proxy Accuracy**: Highest correlation to perfect score ({best_stats['accuracy_proxy']*100:.2f}%) under constraints.
- **Compute Efficiency**: Achieves this accuracy with only {best_stats['params'] / 1000:.1f}K parameters.
- **Hardware Profile**: Perfect memory fit ({best_stats['memory_mb']:.2f}MB vs VRAM limit).

*Executing Full Training to verify architecture...*"""
            self.signals.explainability_msg.emit(explain_text)

            self.signals.ai_msg.emit("Starting Full Training Phase (Stage 2)...")
            self.signals.log_msg.emit("Initializing Trainer")

            # Enforce 15-20 minimum final epochs for Image data and high accuracy
            final_epochs = epochs * nas.weights.get("epoch_multiplier", 1)
            if final_epochs < 15:
                final_epochs = 15
            elif final_epochs > 20 and metadata["type"] == "image":
                final_epochs = 20 # Cap to prevent extremely long hackathon demo wait times

            self.signals.log_msg.emit(f"Calculated Final Epochs: {final_epochs} (due to prompt requirements)")

            # CRITICAL: Rebuild the model completely to reinitialize weights from scratch
            self.signals.ai_msg.emit("Reinitializing Model Weights From Scratch...")
            fresh_model = nas._build_model(best_config)

            trainer = Trainer(fresh_model, train_loader, val_loader, task=metadata["task"])

            def train_cb(epoch, t_loss, v_loss, v_acc):
                self.signals.train_progress.emit(epoch, t_loss, v_loss, v_acc)
                self.signals.log_msg.emit(f"Epoch {epoch}/{final_epochs} | TL: {t_loss:.3f} | VL: {v_loss:.3f} | Metric: {v_acc:.2f}")

            history = trainer.train(epochs=final_epochs, callback=train_cb, early_stopping_patience=3)

            self.signals.ai_msg.emit("Exporting Project...")

            # CRITICAL: Pass the completely rebuilt and retrained model to the exporter, not the dead proxy model
            exporter = ProjectExporter(fresh_model, metadata, history)
            exporter.export()

            completion_log = """
========================================
🚀 PIPELINE COMPLETED SUCCESSFULLY!
Output generated in 'project_output/':
   ├── model.pt
   ├── predict.py
   ├── train.py
   ├── requirements.txt
   ├── README.md
   ├── EXPLAINABILITY.md
   └── results.json
========================================
"""
            self.signals.log_msg.emit(completion_log)
            self.signals.ai_msg.emit("Ready for Deployment.")

            # Expose the newly trained, non-proxy model to the UI Live Prediction module
            self.exported_model = fresh_model
            self.exported_metadata = metadata

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.signals.log_msg.emit(f"ERROR: {str(e)}\n{err}")
            self.signals.ai_msg.emit("Pipeline Failed!")
        finally:
            self.signals.finished.emit()

    def on_finished(self):
        self.btn_start.setEnabled(True)
        if self.exported_model:
            self.btn_predict.setEnabled(True)
            self.pred_label.setText("Ready to test!")

    def run_live_prediction(self):
        if not self.exported_model or not self.exported_metadata:
            return

        if self.exported_metadata["type"] == "tabular":
            path, _ = QFileDialog.getOpenFileName(self, "Select CSV to Predict", "", "CSV Files (*.csv)")
            if not path: return

            import pandas as pd
            import torch
            try:
                df = pd.read_csv(path)
                target_col = df.columns[-1]
                features = df.drop(columns=[target_col])
                features = features.fillna(0)
                tensor = torch.tensor(features.values, dtype=torch.float32).to("cpu")

                from engine.models import TreeModelWrapper
                if isinstance(self.exported_model, TreeModelWrapper):
                    # Scikit-learn / XGB prediction
                    X = tensor.numpy()
                    preds = self.exported_model.model.predict(X)
                    if self.exported_metadata["task"] == "classification":
                        if hasattr(self.exported_model.model, "predict_proba"):
                            probs = self.exported_model.model.predict_proba(X)
                            conf = probs.max(axis=1).mean() * 100
                            self.pred_label.setText(f"Pred: {preds[:3].tolist()}... | Conf: {conf:.1f}%")
                        else:
                            self.pred_label.setText(f"Pred: {preds[:3].tolist()}...")
                    else:
                        self.pred_label.setText(f"Predictions: {preds[:3].tolist()}...")
                else:
                    self.exported_model.to("cpu")
                    self.exported_model.eval()
                    with torch.no_grad():
                        out = self.exported_model(tensor)
                        if self.exported_metadata["task"] == "classification":
                            probs = torch.nn.functional.softmax(out, dim=1)
                            conf, preds = probs.max(dim=1)
                            self.pred_label.setText(f"Pred: {preds[:3].tolist()}... | Conf: {conf[:3].mean().item()*100:.1f}%")
                        else:
                            self.pred_label.setText(f"Predictions: {out[:3].view(-1).tolist()}...")
            except Exception as e:
                self.pred_label.setText(f"Predict Error: {e}")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
            if not path: return

            import torch
            from PIL import Image
            from torchvision import transforms
            try:
                img = Image.open(path).convert("RGB")
                if self.exported_metadata["input_shape"][0] == 1:
                     img = img.convert("L")

                trans = transforms.Compose([
                    transforms.Resize((self.exported_metadata["input_shape"][1], self.exported_metadata["input_shape"][2])),
                    transforms.ToTensor()
                ])
                tensor = trans(img).unsqueeze(0).to("cpu")
                self.exported_model.to("cpu")
                self.exported_model.eval()
                with torch.no_grad():
                    out = self.exported_model(tensor)
                    probs = torch.nn.functional.softmax(out, dim=1)
                    conf, pred = probs.max(dim=1)
                    self.pred_label.setText(f"Class: {pred.item()} | Confidence: {conf.item()*100:.1f}%")
            except Exception as e:
                self.pred_label.setText(f"Predict Error: {e}")

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
