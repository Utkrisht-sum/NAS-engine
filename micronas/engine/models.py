import torch
import torch.nn as nn
from utils.logger import get_logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

logger = get_logger("ModelBuilder")

class TreeModelWrapper:
    """Wrapper to make Scikit/XGB models fit into the PyTorch trainer API conceptually."""
    def __init__(self, model_type, num_classes, task="classification", **kwargs):
        self.task = task
        self.model_type = model_type
        self.kwargs = kwargs

        if task == "classification":
            if model_type == "rf":
                self.model = RandomForestClassifier(**kwargs)
            else:
                self.model = XGBClassifier(**kwargs)
        else:
            if model_type == "rf":
                self.model = RandomForestRegressor(**kwargs)
            else:
                self.model = XGBRegressor(**kwargs)

    def train(self): pass
    def eval(self): pass
    def parameters(self): return []
    def to(self, device): return self
    def __call__(self, x):
        # We handle this manually in trainer for sklearn predict vs predict_proba
        pass

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, task="classification", dropout_rate=0.2):
        super(DynamicLSTM, self).__init__()
        self.task = task
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes if task == "classification" else 1)

    def forward(self, x):
        # x is (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take the last time step
        out = out[:, -1, :]
        return self.fc(out)

class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, task="classification", dropout_rate=0.2):
        super(DynamicGRU, self).__init__()
        self.task = task
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes if task == "classification" else 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class TemporalCNN(nn.Module):
    def __init__(self, input_size, channels, kernel_size, num_classes, task="classification", dropout_rate=0.2):
        super(TemporalCNN, self).__init__()
        self.task = task
        # input is (batch, seq_len, features) -> Conv1d needs (batch, channels, seq_len)
        self.conv = nn.Conv1d(input_size, channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(channels, num_classes if task == "classification" else 1)

    def forward(self, x):
        x = x.transpose(1, 2) # Switch to (batch, features, seq_len)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, task="classification", dropout_rate=0.2):
        super(DynamicMLP, self).__init__()
        self.task = task
        layers = []

        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dynamic regularization
            in_features = out_features

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer
        out_dim = num_classes if task == "classification" else 1
        self.classifier = nn.Linear(in_features, out_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten input in case it's 2D+
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class DynamicCNN(nn.Module):
    def __init__(self, input_shape, conv_layers, fc_layers, num_classes, task="classification", dropout_rate=0.3):
        super(DynamicCNN, self).__init__()
        self.task = task
        self.input_shape = input_shape  # e.g., (3, 32, 32)

        # Robust Block-Based Design: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
        # conv_layers: list of dicts [{'channels': 32, 'kernel_size': 3}]
        layers = []
        in_channels = input_shape[0]

        current_h, current_w = input_shape[1], input_shape[2]

        for idx, config in enumerate(conv_layers):
            out_channels = config['channels']
            k = config['kernel_size']
            p = k // 2

            # Block Dimension Safety Check
            if current_h < k or current_w < k:
                logger.warning(f"Spatial dimension ({current_h}x{current_w}) too small for block {idx}. Repairing to kernel=1.")
                k = 1
                p = 0

            # Conv 1
            layers.append(nn.Conv2d(in_channels, out_channels, k, stride=1, padding=p))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # Conv 2
            layers.append(nn.Conv2d(out_channels, out_channels, k, stride=1, padding=p))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # Pool
            layers.append(nn.MaxPool2d(2, 2))

            in_channels = out_channels
            current_h = current_h // 2
            current_w = current_w // 2

            # Stop adding blocks if dimensions collapse
            if current_h <= 0 or current_w <= 0:
                logger.warning("Spatial dimensions collapsed. Terminating conv blocks early.")
                layers = layers[:-7] # Remove the last complete block to prevent crash
                in_channels = layers[-6].out_channels if len(layers) >= 6 else input_shape[0] # Fallback in_channels
                break

        # Global Average Pooling replaces delicate spatial tracking and massively reduces params
        if not layers:
            # Strong Default Architecture Fallback if everything failed
            logger.warning("All NAS blocks failed or invalid, falling back to strong default CNN architecture.")
            layers = [
                nn.Conv2d(input_shape[0], 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ]
            in_channels = 64

        self.feature_extractor = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        fc_layers_list = []
        in_features = in_channels  # Output channels from last conv after pooling to 1x1
        for out_features in fc_layers:
            fc_layers_list.append(nn.Linear(in_features, out_features))
            fc_layers_list.append(nn.BatchNorm1d(out_features))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(dropout_rate))
            in_features = out_features

        self.fc_block = nn.Sequential(*fc_layers_list)

        out_dim = num_classes if task == "classification" else 1
        self.classifier = nn.Linear(in_features, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    if isinstance(model, TreeModelWrapper): return model.kwargs.get('n_estimators', 100) * 100 # Proxy param count for trees
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_memory_mb(model, input_shape, batch_size=32):
    if isinstance(model, TreeModelWrapper): return 50.0 # Heuristic 50MB for Tree models
    # Very rough estimate for forward pass memory
    params_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # Try a forward pass to estimate activations memory (if possible)
    # Using float32 for dummy
    if len(input_shape) == 1:
        dummy = torch.zeros(batch_size, input_shape[0])
    else:
        dummy = torch.zeros(batch_size, *input_shape)

    try:
        with torch.no_grad():
            out = model(dummy)
        # Assuming intermediate activations roughly equal to output size * layers (very rough heuristic)
        acts_mem = (out.numel() * out.element_size() * 10) / (1024 * 1024)
        total_mem = params_mem + acts_mem
        return total_mem
    except:
        return params_mem * 5 # fallback multiplier
