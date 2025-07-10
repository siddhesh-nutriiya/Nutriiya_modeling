import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import yaml
import logging
import re
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import mlflow.pytorch
mlflow.set_tracking_uri("http://127.0.0.1:5000")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.data_loader import get_image_dataloaders, get_incremental_dataloader, preview_sample_image

# ---------------- LOGGER SETUP ----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('training.log')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---------------- CONFIG LOAD ----------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_model_version(model_seq):
    match = re.search(r'v(\d+)', model_seq.lower())
    return int(match.group(1)) if match else 0

# ---------------- MODEL TRAINING ----------------
def train_model(train_loader, test_loader, val_loader, num_classes, config):
    model_seq = config['model_seq']
    pretrained = config.get('pretrained', False)
    pretrained_path = config.get('pretrained_path', '')
    training_cfg = config['training']
    lr = training_cfg.get('lr', 1e-4)
    num_epochs = training_cfg.get('num_epochs', 20)
    patience = training_cfg.get('patience', 5)
    dropout_prob = config.get('dropout', 0.5)

    device = 'cuda' if training_cfg.get('device', 'cpu') == 'cuda' and torch.cuda.is_available() else 'cpu'

    logger.info(f"Starting training for model {model_seq}")
    logger.info(f"Device: {device}, Pretrained: {pretrained}, LR: {lr}, Epochs: {num_epochs}, Dropout: {dropout_prob}")

    class CustomResNet(nn.Module):
        def __init__(self, base_model, num_classes, dropout_prob):
            super().__init__()
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.dropout = nn.Dropout(dropout_prob)
            self.fc = nn.Linear(base_model.fc.in_features, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    base_model = models.resnet18(weights=None if pretrained else 'IMAGENET1K_V1')
    model = CustomResNet(base_model, num_classes, dropout_prob).to(device)

    if pretrained and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        logger.info(f"Loaded pretrained weights from: {pretrained_path}")
    elif pretrained:
        logger.warning(f"Pretrained path {pretrained_path} not found. Starting with random weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    epochs_no_improve = 0
    model_version = extract_model_version(model_seq)
    model_folder = os.path.join("models", f"model_v{model_version}")
    os.makedirs(model_folder, exist_ok=True)    

    with mlflow.start_run():
        mlflow.log_param("model_seq", model_seq)
        mlflow.log_param("lr", lr)
        mlflow.log_param("dropout", dropout_prob)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("device", device)

        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            running_loss, running_correct, running_total = 0.0, 0, 0

            loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, labels) in loop:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)

            epoch_loss = running_loss / running_total
            epoch_acc = running_correct / running_total
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch)

            eta = (time.time() - start_time) * (num_epochs - (epoch + 1))
            eta_min, eta_sec = divmod(int(eta), 60)

            logger.info(f"[Train] Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | ETA: {eta_min}m {eta_sec}s")

            if val_loader:
                model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        preds = outputs.argmax(dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
                val_acc = val_correct / val_total
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                logger.info(f"[Val] Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
            else:
                val_acc = epoch_acc

            # --- Save every 5th epoch ---
            if (epoch + 1) % 5 == 0:
                epoch_path = os.path.join(model_folder, f"epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), epoch_path)
                logger.info(f"Saved checkpoint at: {epoch_path}")

            # --- Save Best Model ---
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                best_path = os.path.join(model_folder, "best.pth")
                torch.save(model.state_dict(), best_path)
                mlflow.pytorch.log_model(model, "best_model")
                logger.info(f"Best model saved at: {best_path}")
            else:
                epochs_no_improve += 1

            # --- Always save latest ---
            latest_path = os.path.join(model_folder, "latest.pth")
            torch.save(model.state_dict(), latest_path)

            # --- Stop if acc too high or early stop ---
            if val_acc >= 0.95 or epoch_acc >= 0.95:
                logger.info(f"Accuracy threshold reached (Train: {epoch_acc:.4f}, Val: {val_acc:.4f}). Stopping training.")
                break

            if epochs_no_improve >= patience:
                logger.warning("Early stopping triggered.")
                break

        logger.info(f"Training complete. Best Val Accuracy: {best_acc:.4f}")
        mlflow.log_metric("best_val_acc", best_acc)

# ----------------- MAIN --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--val_split', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--incremental', action='store_true', help='Enable incremental training')
    parser.add_argument('--round', type=int, default=1, help='Current incremental round number')
    parser.add_argument('--run_name', type=str, default=None, help='Name for MLflow run')  

    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg = config['data']

    if args.incremental:
        train_loader = get_incremental_dataloader(
            base_dir=data_cfg['base_dir'],
            current_round=args.round,
            batch_size=data_cfg.get('batch_size', 128),
            img_size=data_cfg.get('img_size', 224),
            num_workers=data_cfg.get('num_workers', 4)
        )
        test_loader = val_loader = None
    else:
        train_loader, test_loader, val_loader = get_image_dataloaders(
            data_dir=data_cfg['data_dir'],
            batch_size=data_cfg.get('batch_size', 128),
            img_size=data_cfg.get('img_size', 224),
            num_workers=data_cfg.get('num_workers', 4),
            val_split=args.val_split
        )

    dataset = train_loader.dataset
    num_classes = len(dataset.dataset.classes) if hasattr(dataset, 'dataset') else len(dataset.classes)

    preview_sample_image(train_loader)
    train_model(train_loader, test_loader, val_loader, num_classes, config)
