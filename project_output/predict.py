import torch
import argparse
import pandas as pd
from PIL import Image
from torchvision import transforms
import json
import os

# We package the dynamic model architectures directly into the predict script
# to ensure it is fully standalone without depending on the engine codebase.

import torch.nn as nn

class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, task="classification"):
        super(DynamicMLP, self).__init__()
        layers = []
        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_features = out_features
        self.feature_extractor = nn.Sequential(*layers)
        out_dim = num_classes if task == "classification" else 1
        self.classifier = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        return self.classifier(x)

class DynamicCNN(nn.Module):
    def __init__(self, input_shape, conv_layers, fc_layers, num_classes, task="classification"):
        super(DynamicCNN, self).__init__()
        layers = []
        in_channels = input_shape[0]
        current_h, current_w = input_shape[1], input_shape[2]
        for idx, config in enumerate(conv_layers):
            out_channels = config['channels']
            k = config['kernel_size']
            s = config.get('stride', 1)
            p = config.get('padding', k // 2)
            if current_h < k or current_w < k:
                k, p = 1, 0
            layers.append(nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            current_h = (current_h - k + 2*p) // s + 1
            current_w = (current_w - k + 2*p) // s + 1
            current_h, current_w = current_h // 2, current_w // 2
            if current_h <= 0 or current_w <= 0:
                layers = layers[:-4]
                break
        self.feature_extractor = nn.Sequential(*layers)
        dummy_input = torch.zeros(1, *input_shape)
        try:
            with torch.no_grad():
                dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        except Exception:
            flattened_size = 1
        fc_layers_list = []
        in_features = flattened_size
        for out_features in fc_layers:
            fc_layers_list.append(nn.Linear(in_features, out_features))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(0.3))
            in_features = out_features
        self.fc_block = nn.Sequential(*fc_layers_list)
        out_dim = num_classes if task == "classification" else 1
        self.classifier = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return self.classifier(x)

def load_model():
    print("Loading architecture memory...")
    with open('architecture_memory.json', 'r') as f:
        memory = json.load(f)

    config = memory['best_config']
    input_shape = memory['input_shape']
    # If the user used mock CSV, num_classes won't be in memory cleanly,
    # but we will default to 2 for hackathon MVP or read from state dict.

    # We load state dict directly and infer classes from classifier weight
    state_dict = torch.load('model.pt', map_location='cpu')
    num_classes = state_dict['classifier.weight'].shape[0]

    print("Reconstructing architecture...")
    if config['type'] == 'mlp':
        model = DynamicMLP(input_size=input_shape[0], hidden_layers=config['hidden_layers'], num_classes=num_classes)
    else:
        model = DynamicCNN(input_shape=input_shape, conv_layers=config['conv_layers'], fc_layers=config['fc_layers'], num_classes=num_classes)

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")
    return model, memory['dataset_type']

def predict(input_data):
    model, dataset_type = load_model()
    print(f"Predicting on: {input_data}")

    if dataset_type == "tabular":
        df = pd.read_csv(input_data)
        # Assuming last column is missing/target drop
        tensor = torch.tensor(df.values, dtype=torch.float32)
        with torch.no_grad():
            preds = model(tensor)
            print("Predictions:\n", preds)
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = Image.open(input_data).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            preds = model(tensor)
            print("Prediction Logits:\n", preds)
            print("Predicted Class:\n", torch.argmax(preds, dim=1).item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', type=str, help='Path to input file (CSV/Image)')
    args = parser.parse_args()
    predict(args.input_data)
