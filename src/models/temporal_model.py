"""
PyTorch Temporal Deep Learning Models for Dynamic Indian Sign Language Recognition.
Supports Bidirectional LSTM (BiLSTM) and GRU architectures over landmark sequences.
"""

from typing import Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_optimal_device() -> torch.device:
    """Detects and returns MPS (Apple Silicon), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ISLBiLSTM(nn.Module):
    """
    Bidirectional LSTM with Dropout and Batch Normalization for sequence classification.
    Input shape: (batch_size, seq_len=30, input_dim=126)
    Output shape: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully Connected Classifier
        # Bidirectional doubles the hidden size (hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)

        # Average pooling across time dimension + last step representation
        time_avg = torch.mean(lstm_out, dim=1) # (batch, hidden_size * 2)

        out = self.fc1(time_avg)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


def save_checkpoint(model: nn.Module, path: str, class_names: list, metadata: dict = None):
    """Saves model weights along with class mappings and metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_classes": model.num_classes,
        "class_names": class_names,
        "metadata": metadata or {}
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device = None) -> Tuple[ISLBiLSTM, list]:
    """Loads model weights and returns (model, class_names)."""
    if device is None:
        device = get_optimal_device()

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ISLBiLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["class_names"]
