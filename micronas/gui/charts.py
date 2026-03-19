import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout

class ChartsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # Plot 1: NAS Progress (Generations vs Fitness)
        self.fig_nas = Figure(figsize=(5, 3))
        self.canvas_nas = FigureCanvas(self.fig_nas)
        self.ax_nas = self.fig_nas.add_subplot(111)
        self.ax_nas.set_title("NAS Progress")
        self.ax_nas.set_xlabel("Generation")
        self.ax_nas.set_ylabel("Fitness")

        # Plot 2: Training Progress (Epochs vs Loss)
        self.fig_train = Figure(figsize=(5, 3))
        self.canvas_train = FigureCanvas(self.fig_train)
        self.ax_train = self.fig_train.add_subplot(111)
        self.ax_train.set_title("Training Loss")
        self.ax_train.set_xlabel("Epoch")
        self.ax_train.set_ylabel("Loss")

        # Plot 3: Pareto Frontier (Params vs Accuracy proxy)
        self.fig_pareto = Figure(figsize=(5, 3))
        self.canvas_pareto = FigureCanvas(self.fig_pareto)
        self.ax_pareto = self.fig_pareto.add_subplot(111)
        self.ax_pareto.set_title("Pareto Frontier (NAS)")
        self.ax_pareto.set_xlabel("Params (Millions)")
        self.ax_pareto.set_ylabel("Fitness")

        self.layout.addWidget(self.canvas_nas)
        self.layout.addWidget(self.canvas_train)
        self.layout.addWidget(self.canvas_pareto)

        # Initialize data
        self.nas_gens = []
        self.nas_fitness = []

        self.train_epochs = []
        self.train_losses = []
        self.val_losses = []

    def update_nas_chart(self, generation, best_fitness):
        self.nas_gens.append(generation)
        self.nas_fitness.append(best_fitness)
        self.ax_nas.clear()
        self.ax_nas.plot(self.nas_gens, self.nas_fitness, marker='o', color='blue')
        self.ax_nas.set_title("NAS Progress")
        self.ax_nas.set_xlabel("Generation")
        self.ax_nas.set_ylabel("Fitness")
        self.canvas_nas.draw()

    def update_train_chart(self, epoch, train_loss, val_loss):
        self.train_epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.ax_train.clear()
        self.ax_train.plot(self.train_epochs, self.train_losses, label="Train Loss", color='red')
        self.ax_train.plot(self.train_epochs, self.val_losses, label="Val Loss", color='orange')
        self.ax_train.legend()
        self.ax_train.set_title("Training Loss")
        self.ax_train.set_xlabel("Epoch")
        self.ax_train.set_ylabel("Loss")
        self.canvas_train.draw()

    def update_pareto_chart(self, population_data):
        self.ax_pareto.clear()
        params = [ind["params"] / 1e6 for ind in population_data]
        fitness = [ind["fitness"] for ind in population_data]

        self.ax_pareto.scatter(params, fitness, color='green', alpha=0.6)
        self.ax_pareto.set_title("Pareto Frontier")
        self.ax_pareto.set_xlabel("Params (Millions)")
        self.ax_pareto.set_ylabel("Fitness")
        self.canvas_pareto.draw()
