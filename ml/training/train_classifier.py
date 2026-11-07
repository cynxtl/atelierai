"""Training script for Auxiliary Fashion-MNIST CNN Classifier."""

import argparse
import json
import logging
import time

import torch
from torch import nn, optim

from ml.configs.config import FASHION_CLASSES, ClassifierConfig
from ml.data.fashion_mnist import get_fashion_mnist_dataloaders
from ml.models.classifier import FashionCNNClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_classifier(config: ClassifierConfig):
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load data
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        for_model="classifier",
        batch_size=config.batch_size,
        seed=config.seed,
        val_split=0.1,
    )

    model = FashionCNNClassifier(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()
    logger.info(f"Starting training for {config.epochs} epochs...")

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        scheduler.step(epoch_val_acc)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        logger.info(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "config": config.__dict__,
            }, config.checkpoint_path)
            logger.info(f" Saved new best model checkpoint to {config.checkpoint_path}")

    # Evaluate on final test set
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_correct = 0
    test_total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                lbl = labels[i].item()
                class_total[lbl] += 1
                if predicted[i] == lbl:
                    class_correct[lbl] += 1

    overall_test_acc = test_correct / test_total
    class_accuracies = {
        FASHION_CLASSES[i]: round(class_correct[i] / class_total[i], 4) for i in range(10)
    }

    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {elapsed_time}s. Overall Test Accuracy: {overall_test_acc * 100:.2f}%")
    for cls, acc in class_accuracies.items():
        logger.info(f"  - {cls:15s}: {acc * 100:.2f}%")

    # Save metrics
    metrics = {
        "overall_test_accuracy": round(overall_test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "class_accuracies": class_accuracies,
        "history": history,
        "training_time_seconds": elapsed_time,
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
    }

    with open(config.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved classifier metrics to {config.metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Auxiliary Fashion-MNIST Classifier")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = ClassifierConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    train_classifier(cfg)
