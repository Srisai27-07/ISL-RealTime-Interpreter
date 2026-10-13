"""
PyTorch Model Training Script for ISL BiLSTM Gesture Classifier.
Loads preprocessed landmark sequences from `data/processed/`, trains the model,
and outputs evaluation metrics and checkpoint weights.

Bonus Feature: Includes `--generate-mock` flag so the team can test the end-to-end
pipeline and UI before real dataset collection is complete!

Usage:
    python scripts/03_train_model.py
    python scripts/03_train_model.py --generate-mock   # Generates dummy data for pipeline smoke test
"""

import os
import sys
import glob
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append(".")

from src.models.temporal_model import ISLBiLSTM, save_checkpoint, get_optimal_device


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


class ISLSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def generate_synthetic_data(classes: list, samples_per_class: int = 15, seq_len: int = 30):
    """Creates synthetic landmark sequences to test the training and inference pipeline immediately."""
    print("[INFO] Generating synthetic landmark sequences for rapid pipeline testing...")
    for class_name in classes:
        folder = os.path.join("data/processed", class_name)
        os.makedirs(folder, exist_ok=True)
        # Class-specific pattern with random jitter
        class_seed = abs(hash(class_name)) % 1000
        np.random.seed(class_seed)
        base_pattern = np.random.randn(seq_len, 126).astype(np.float32) * 0.1

        for i in range(samples_per_class):
            jitter = np.random.normal(0, 0.02, size=(seq_len, 126)).astype(np.float32)
            sample = base_pattern + jitter
            out_file = os.path.join(folder, f"mock_sample_{i}.npy")
            np.save(out_file, sample)
    print("[SUCCESS] Mock dataset generated in data/processed/")


def load_dataset(processed_dir: str, classes: list):
    sequences = []
    labels = []

    for label_idx, class_name in enumerate(classes):
        class_folder = os.path.join(processed_dir, class_name)
        if not os.path.exists(class_folder):
            continue
        npy_files = glob.glob(os.path.join(class_folder, "*.npy"))
        for npy_path in npy_files:
            arr = np.load(npy_path)
            # Ensure shape is (30, 126)
            if arr.shape[0] == 30 and arr.shape[1] == 126:
                sequences.append(arr)
                labels.append(label_idx)

    if not sequences:
        return None, None

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-mock", action="store_true", help="Generate mock samples if dataset is empty")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    config = load_config()
    classes = config["classes"]
    processed_dir = "data/processed"

    if args.generate_mock:
        generate_synthetic_data(classes)

    X, y = load_dataset(processed_dir, classes)

    if X is None or len(X) < len(classes) * 2:
        print("[WARNING] Insufficient data found in data/processed/.")
        print("To generate mock data for an immediate pipeline test, run:")
        print("    python scripts/03_train_model.py --generate-mock")
        return

    print(f"Loaded {len(X)} total sequences across {len(classes)} classes.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(ISLSequenceDataset(X_train, y_train), batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(ISLSequenceDataset(X_val, y_val), batch_size=config["model"]["batch_size"], shuffle=False)

    device = get_optimal_device()
    print(f"Training on device: {device}")

    model = ISLBiLSTM(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=len(classes),
        dropout=config["model"]["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-4)

    best_val_acc = 0.0
    checkpoint_path = config["model"]["weights_path"]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        train_acc = correct / total

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                outputs = model(seqs)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == targets).sum().item()
                v_total += len(targets)

        val_acc = v_correct / max(v_total, 1)

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, checkpoint_path, classes, metadata={"val_acc": val_acc})

    print(f"\n[DONE] Best model saved to {checkpoint_path} with Val Acc: {best_val_acc*100:.1f}%")


if __name__ == "__main__":
    train()
