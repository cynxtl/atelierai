"""Fine-tuning script for DistilBERT on natural language clothing prompts."""

import argparse
import json
import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ml.configs.config import FASHION_CLASSES, DistilBERTConfig
from ml.models.distilbert_classifier import (
    FashionTextDataset,
    generate_synthetic_fashion_text_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_distilbert(config: DistilBERTConfig):
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    logger.info(f"Loading base tokenizer and model from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
    ).to(device)

    dataset_splits = generate_synthetic_fashion_text_dataset()
    train_texts, train_labels = zip(*dataset_splits["train"])
    val_texts, val_labels = zip(*dataset_splits["val"])
    test_texts, test_labels = zip(*dataset_splits["test"])

    logger.info(f"Dataset sizes: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    train_dataset = FashionTextDataset(list(train_texts), list(train_labels), tokenizer)
    val_dataset = FashionTextDataset(list(val_texts), list(val_labels), tokenizer)
    test_dataset = FashionTextDataset(list(test_texts), list(test_labels), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start_time = time.time()
    logger.info(f"Fine-tuning DistilBERT for {config.epochs} epochs...")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            config.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)
            logger.info(f" Saved new best model to {config.output_dir}")

    # Evaluate on final test set
    model = AutoModelForSequenceClassification.from_pretrained(config.output_dir).to(device)
    model.eval()

    test_correct = 0
    test_total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()

            for i in range(labels.size(0)):
                lbl = labels[i].item()
                class_total[lbl] += 1
                if preds[i] == lbl:
                    class_correct[lbl] += 1

    overall_test_acc = test_correct / test_total
    class_accuracies = {
        FASHION_CLASSES[i]: round(class_correct[i] / max(1, class_total[i]), 4) for i in range(10)
    }

    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"DistilBERT fine-tuning finished in {elapsed_time}s. Test Accuracy: {overall_test_acc * 100:.2f}%")

    metrics = {
        "overall_test_accuracy": round(overall_test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "class_accuracies": class_accuracies,
        "history": history,
        "training_time_seconds": elapsed_time,
        "device": str(device),
        "epochs": config.epochs,
        "learning_rate": config.lr,
    }

    with open(config.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved DistilBERT metrics to {config.metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Clothing Text Classification")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = DistilBERTConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    train_distilbert(cfg)
