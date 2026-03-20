import random
import time
import json
import os
import torch
from utils.logger import get_logger
from engine.models import DynamicMLP, DynamicCNN, DynamicLSTM, DynamicGRU, TemporalCNN, TreeModelWrapper, count_parameters, estimate_memory_mb
from engine.prompt_parser import PromptParser

logger = get_logger("NASEngine")

class NASEngine:
    def __init__(self, metadata, train_loader, val_loader, prompt=""):
        self.metadata = metadata
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.prompt = prompt

        parser = PromptParser()
        self.weights = parser.parse(prompt)

        self.population = []
        self.history = []
        self.pareto_front = []
        self.failures = [] # Failure-aware system

        self.best_model = None
        self.best_config = None

    def _sample_tabular_config(self):
        # Diverse Tabular Architectures including Tree Models
        num_samples = self.metadata.get("num_samples", 1000)

        # Base templates
        templates = [
            {
                "type": "rf",
                "name": "Random Forest",
                "n_estimators": random.choice([50, 100, 200]),
                "max_depth": random.choice([10, 20, None])
            },
            {
                "type": "xgb",
                "name": "XGBoost",
                "n_estimators": random.choice([100, 200, 300]),
                "learning_rate": random.choice([0.01, 0.1, 0.2]),
                "max_depth": random.choice([3, 5, 7])
            },
            {
                "type": "mlp",
                "name": "Shallow MLP",
                "hidden_layers": [32]
            }
        ]

        if num_samples > 500:
            templates.extend([
                {
                    "type": "mlp",
                    "name": "Wide MLP",
                    "hidden_layers": [128, 128]
                },
                {
                    "type": "mlp",
                    "name": "Balanced MLP",
                    "hidden_layers": [128, 64, 32]
                }
            ])

        if num_samples > 5000:
            templates.append({
                "type": "mlp",
                "name": "Deep MLP",
                "hidden_layers": [64, 64, 128, 128]
            })
            templates[0]["n_estimators"] = random.choice([300, 500]) # Deeper RF
            templates[1]["n_estimators"] = random.choice([300, 500]) # Deeper XGB

        return random.choice(templates)

    def _sample_cnn_config(self):
        # Using Provided Predefined CNN Templates (Small, Medium, Deep) instead of random layers
        num_samples = self.metadata.get("num_samples", 1000)

        templates = [
            {
                "type": "cnn",
                "name": "Small CNN",
                "conv_layers": [{"channels": 32, "kernel_size": 3}, {"channels": 64, "kernel_size": 3}],
                "fc_layers": [128]
            }
        ]

        if num_samples > 1000:
            templates.append({
                "type": "cnn",
                "name": "Medium CNN",
                "conv_layers": [{"channels": 32, "kernel_size": 3}, {"channels": 64, "kernel_size": 3}, {"channels": 128, "kernel_size": 3}],
                "fc_layers": [256]
            })

        if num_samples > 10000:
            templates.append({
                "type": "cnn",
                "name": "Deep CNN",
                "conv_layers": [{"channels": 64, "kernel_size": 3}, {"channels": 128, "kernel_size": 3}, {"channels": 256, "kernel_size": 3}, {"channels": 256, "kernel_size": 3}],
                "fc_layers": [512, 128]
            })

        return random.choice(templates)

    def _sample_sequence_config(self):
        depth = random.randint(1, 3)
        hidden = random.choice([32, 64, 128])
        templates = [
            {
                "type": "lstm",
                "name": f"LSTM (Layers: {depth}, Hidden: {hidden})",
                "hidden_size": hidden,
                "num_layers": depth
            },
            {
                "type": "gru",
                "name": f"GRU (Layers: {depth}, Hidden: {hidden})",
                "hidden_size": hidden,
                "num_layers": depth
            },
            {
                "type": "tcnn",
                "name": f"Temporal CNN (Channels: {hidden})",
                "channels": hidden,
                "kernel_size": random.choice([3, 5])
            }
        ]
        return random.choice(templates)

    def _build_model(self, config):
        try:
            dropout_rate = self.weights.get("dropout_rate", 0.2)
            if config["type"] == "mlp":
                model = DynamicMLP(
                    input_size=self.metadata["input_shape"][0],
                    hidden_layers=config["hidden_layers"],
                    num_classes=self.metadata["num_classes"],
                    task=self.metadata["task"],
                    dropout_rate=dropout_rate
                )
            elif config["type"] == "gru":
                model = DynamicGRU(
                    input_size=self.metadata["input_shape"][1] if len(self.metadata["input_shape"]) > 1 else 1,
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    num_classes=self.metadata["num_classes"],
                    task=self.metadata["task"],
                    dropout_rate=dropout_rate
                )
            elif config["type"] == "tcnn":
                model = TemporalCNN(
                    input_size=self.metadata["input_shape"][1] if len(self.metadata["input_shape"]) > 1 else 1,
                    channels=config["channels"],
                    kernel_size=config["kernel_size"],
                    num_classes=self.metadata["num_classes"],
                    task=self.metadata["task"],
                    dropout_rate=dropout_rate
                )
            elif config["type"] == "cnn":
                model = DynamicCNN(
                    input_shape=self.metadata["input_shape"],
                    conv_layers=config["conv_layers"],
                    fc_layers=config["fc_layers"],
                    num_classes=self.metadata["num_classes"],
                    task=self.metadata["task"],
                    dropout_rate=dropout_rate
                )
            elif config["type"] == "lstm":
                model = DynamicLSTM(
                    input_size=self.metadata["input_shape"][1] if len(self.metadata["input_shape"]) > 1 else 1,
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    num_classes=self.metadata["num_classes"],
                    task=self.metadata["task"],
                    dropout_rate=dropout_rate
                )
            elif config["type"] in ["rf", "xgb"]:
                # Tree models are instantiated via wrapper
                model_params = {k:v for k,v in config.items() if k not in ["type", "name"]}
                model = TreeModelWrapper(
                    model_type=config["type"],
                    num_classes=self.metadata["num_classes"],
                    task=self.metadata["task"],
                    **model_params
                )
            return model
        except Exception as e:
            logger.error(f"Failed to build model from config {config}: {e}")
            self.failures.append(config)
            return None

    def _evaluate_fitness(self, model, config, device="cpu"):
        # F(A) = validation_accuracy (Strict Training-Based Fitness)
        # CRITICAL FAILSAFE: If a model crashes during training (e.g. memory, shape), it gets fitness = -1 so NAS naturally filters it out.

        val_acc = 0.0
        train_loss = float('inf')
        latency = 999.0
        params = count_parameters(model)
        memory_mb = estimate_memory_mb(model, self.metadata["input_shape"])

        try:
            from engine.trainer import Trainer
            import logging
            old_level = logging.getLogger("Trainer").level
            logging.getLogger("Trainer").setLevel(logging.CRITICAL) # suppress logs

            # 1. Latency (Estimate with forward pass first to ensure model is structurally valid)
            model.to(device)
            dummy_input = next(iter(self.train_loader))[0][:4].to(device)
            start = time.time()
            with torch.no_grad():
                model(dummy_input)
            latency = (time.time() - start) * 1000 # ms

            # 2. Real Validation Training (Stage 1)
            # We enforce exactly 2 epochs of training for the proxy evaluation.
            proxy_trainer = Trainer(model, self.train_loader, self.val_loader, task=self.metadata["task"])

            # Using standard training loop
            history = proxy_trainer.train(epochs=2, early_stopping_patience=2)

            val_acc = history['val_acc'][-1] if len(history['val_acc']) > 0 else 0.0
            # Track train_acc to detect over/under fitting
            train_loss = history['train_loss'][-1] if len(history['train_loss']) > 0 else float('inf')

        except Exception as e:
            logger.warning(f"Training evaluation failed for {config.get('name', 'Model')}: {e}")
            val_acc = -1.0 # Guarantee rejection by Evolutionary Sort
            train_loss = float('inf')
        finally:
            try:
                logging.getLogger("Trainer").setLevel(old_level)
                model.to("cpu") # Free VRAM
            except:
                pass

        # VALIDATION-FIRST LOGIC: Always prioritize val_acc over train_acc or zero-cost guesses
        fitness = val_acc

        return {
            "fitness": fitness,
            "accuracy_proxy": max(0, val_acc / 100.0), # Kept for UI compatibility (e.g. 0.95), clamp negative
            "train_loss": train_loss, # Used to detect overfitting
            "params": params,
            "latency_ms": latency,
            "memory_mb": memory_mb,
            "config": config,
            "model": model
        }

    def run_search(self, population_size=10, generations=3, max_params=1e6):
        logger.info(f"Starting NAS | Pop: {population_size}, Gens: {generations}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running NAS on {device}")

        # Initialize Population
        for _ in range(population_size):
            if self.metadata["type"] == "tabular":
                config = self._sample_tabular_config()
            elif self.metadata["type"] == "image":
                config = self._sample_cnn_config()
            else:
                config = self._sample_sequence_config()

            model = self._build_model(config)

            if model and count_parameters(model) < max_params:
                eval_data = self._evaluate_fitness(model, config, device)
                self.population.append(eval_data)

        # Filter out crashed models
        self.population = [p for p in self.population if p["fitness"] > -1]

        # Handle case where all models exceeded max_params or crashed (Fallback)
        if not self.population:
            logger.warning("All initial models failed or exceeded max params. Injecting failsafe fallback.")
            if self.metadata["type"] == "tabular":
                config = self._sample_tabular_config()
            elif self.metadata["type"] == "image":
                config = self._sample_cnn_config()
            else:
                config = self._sample_sequence_config()
            model = self._build_model(config)
            eval_data = self._evaluate_fitness(model, config, device)
            self.population.append(eval_data)

        # Evolutionary Loop
        for gen in range(generations):
            logger.info(f"--- Generation {gen+1}/{generations} ---")

            # Sort by fitness
            self.population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
            self.history.append([ind["fitness"] for ind in self.population])

            # Keep top 50%
            parents = self.population[:max(1, population_size//2)]

            # Mutate to fill population
            next_gen = parents.copy()
            while len(next_gen) < population_size:
                parent = random.choice(parents)["config"]
                child_config = self._mutate(parent)

                # Check failure memory
                if child_config in self.failures:
                    logger.info("Avoided known failed architecture.")
                    continue

                child_model = self._build_model(child_config)
                if child_model and count_parameters(child_model) < max_params:
                    eval_data = self._evaluate_fitness(child_model, child_config, device)
                    next_gen.append(eval_data)

            self.population = next_gen

        # Final Selection
        self.population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        best = self.population[0]
        self.best_model = best["model"]
        self.best_config = best["config"]

        logger.info(f"NAS Completed. Best Fitness: {best['fitness']:.4f}")
        logger.info(f"Best Config: {self.best_config}")

        self._save_memory(best)

        return self.best_model, self.best_config, self.population

    def _mutate(self, config, parent_stats=None):
        import copy
        new_config = copy.deepcopy(config)

        # Analyze performance if available to do targeted mutation
        is_overfitting = False
        is_underfitting = False

        if parent_stats:
            val_acc = parent_stats.get("accuracy_proxy", 0) * 100.0
            train_loss = parent_stats.get("train_loss", float('inf'))

            # Explicit rule: train_acc >> val_acc
            # We estimate train_acc via an exponential heuristic from train_loss.
            train_acc_est = min(99.0, (1.0 / (train_loss + 0.01)) * 10.0 + val_acc)

            if (train_acc_est - val_acc) > 10.0:
                is_overfitting = True
                logger.info(f"Overfitting detected (Train Acc: {train_acc_est:.2f}%, Val Acc: {val_acc:.2f}%). Adapting config.")
                # Adaptive Dropout: Increase dropout dynamically to combat memorization
                self.weights["dropout_rate"] = min(0.5, self.weights.get("dropout_rate", 0.2) + 0.1)

            # Explicit rule: train_acc and val_acc both low
            elif train_acc_est < 60 and val_acc < 60:
                is_underfitting = True
                logger.info(f"Underfitting detected (Train Acc: {train_acc_est:.2f}%, Val Acc: {val_acc:.2f}%). Adapting config.")
                # Train longer: Increment the epoch multiplier automatically
                self.weights["epoch_multiplier"] = self.weights.get("epoch_multiplier", 1) + 1

        if new_config["type"] == "mlp":
            if is_overfitting and len(new_config["hidden_layers"]) > 1:
                logger.info("Overfitting detected. Mutating to a smaller MLP.")
                new_config["hidden_layers"].pop()
            elif is_underfitting and len(new_config["hidden_layers"]) < 5:
                logger.info("Underfitting detected. Mutating to a deeper MLP.")
                new_config["hidden_layers"].append(random.choice([64, 128, 256]))
            else:
                # Random mutation
                if random.random() < 0.3 and len(new_config["hidden_layers"]) < 5:
                    new_config["hidden_layers"].append(random.choice([16, 32, 64]))
                elif random.random() < 0.3 and len(new_config["hidden_layers"]) > 1:
                    new_config["hidden_layers"].pop()
                else:
                    idx = random.randint(0, len(new_config["hidden_layers"]) - 1)
                    new_config["hidden_layers"][idx] = random.choice([16, 32, 64, 128])
        elif new_config["type"] == "cnn":
            if is_overfitting and len(new_config["conv_layers"]) > 2:
                logger.info("Overfitting detected. Mutating to a shallower CNN.")
                new_config["conv_layers"].pop()
            elif is_underfitting and len(new_config["conv_layers"]) < 5:
                logger.info("Underfitting detected. Mutating to a deeper CNN.")
                new_config["conv_layers"].append({"channels": random.choice([64, 128]), "kernel_size": 3})
            else:
                # Random mutation
                if random.random() < 0.3 and len(new_config["conv_layers"]) < 5:
                    new_config["conv_layers"].append({"channels": random.choice([16,32]), "kernel_size": 3})
                elif random.random() < 0.3 and len(new_config["conv_layers"]) > 2: # Keep at least 2 for block
                    new_config["conv_layers"].pop()
                else:
                    idx = random.randint(0, len(new_config["conv_layers"]) - 1)
                    new_config["conv_layers"][idx]["channels"] = random.choice([16, 32, 64])
        elif new_config["type"] in ["rf", "xgb"]:
            # Trees: simply change hyperparameters slightly
            if is_overfitting:
                new_config["max_depth"] = max(3, (new_config.get("max_depth") or 10) - 2)
            elif is_underfitting:
                new_config["n_estimators"] += 50
            else:
                new_config["n_estimators"] += random.choice([-50, 0, 50])
                new_config["n_estimators"] = max(10, new_config["n_estimators"])
        elif new_config["type"] in ["lstm", "gru"]:
            if is_overfitting and new_config["num_layers"] > 1:
                new_config["num_layers"] -= 1
            elif is_underfitting:
                new_config["hidden_size"] = min(256, new_config["hidden_size"] * 2)
            else:
                new_config["hidden_size"] = random.choice([32, 64, 128])
        elif new_config["type"] == "tcnn":
            if is_overfitting:
                new_config["channels"] = max(16, new_config["channels"] // 2)
            elif is_underfitting:
                new_config["channels"] = min(256, new_config["channels"] * 2)
            else:
                new_config["kernel_size"] = random.choice([3, 5, 7])

        return new_config

    def _save_memory(self, best_eval):
        os.makedirs("project_output", exist_ok=True)
        memory = {
            "dataset_type": self.metadata["type"],
            "input_shape": self.metadata["input_shape"],
            "best_config": best_eval["config"],
            "performance": {
                "fitness": best_eval["fitness"],
                "params": best_eval["params"],
                "memory_mb": best_eval["memory_mb"]
            }
        }
        with open("project_output/architecture_memory.json", "w") as f:
            json.dump(memory, f, indent=4)
        logger.info("Saved architecture memory to disk.")
