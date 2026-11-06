"""
ClinScribe AI — NER Training Module
=====================================
Fine-tunes BioBERT / ClinicalBERT for clinical Named Entity Recognition
using token classification with BIO tagging.
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ner.dataset import (
    ClinicalNERDataset,
    get_label_maps,
    DEFAULT_LABEL_LIST,
    prepare_hf_ner_dataset,
    load_bc5cdr_dataset,
)
from src.utils.metrics import compute_ner_metrics, print_metrics_report


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: dict):
    """
    Initialize BioBERT model for token classification.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (model, tokenizer, label_list).
    """
    model_name = config["model"]["name"]
    label_list = config["data"].get("label_list", DEFAULT_LABEL_LIST)
    label2id, id2label = get_label_maps(label_list)

    print(f"📦 Loading NER model: {model_name}")
    print(f"   Labels: {len(label_list)} ({', '.join(label_list[:5])}...)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
    )

    return model, tokenizer, label_list


def create_compute_metrics(label_list: list):
    """
    Create a metrics function for the Trainer.

    Uses seqeval for entity-level evaluation.
    """

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Convert IDs back to label strings, filtering -100
        id2label = {i: label for i, label in enumerate(label_list)}

        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_labels = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id != -100:
                    true_seq.append(id2label.get(label_id, "O"))
                    pred_seq_labels.append(id2label.get(pred_id, "O"))

            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)

        metrics = compute_ner_metrics(pred_labels, true_labels)
        # Remove the classification_report string for Trainer compatibility
        return {k: v for k, v in metrics.items() if k != "classification_report"}

    return compute_metrics


def train(config_path: str):
    """
    Main training function for clinical NER.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)

    # ── Setup Model ──────────────────────────────────────────────────────────
    model, tokenizer, label_list = setup_model_and_tokenizer(config)

    # ── Load Dataset ─────────────────────────────────────────────────────────
    print("📂 Loading clinical NER dataset...")

    data_config = config["data"]
    max_seq_length = data_config.get("max_seq_length", 512)

    # Try loading from local files first
    train_path = data_config.get("train_dataset")
    eval_path = data_config.get("eval_dataset")

    if train_path and os.path.exists(train_path):
        train_dataset = ClinicalNERDataset(
            data_path=os.path.join(train_path, "data.json"),
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=max_seq_length,
        )
        eval_dataset = ClinicalNERDataset(
            data_path=os.path.join(eval_path, "data.json"),
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=max_seq_length,
        )
    else:
        # Fallback: Load from HuggingFace
        print("   Loading BC5CDR from HuggingFace...")
        from datasets import load_dataset

        raw_dataset = load_dataset("ncbi_disease")

        train_dataset = prepare_hf_ner_dataset(
            raw_dataset["train"],
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=max_seq_length,
        )
        eval_dataset = prepare_hf_ner_dataset(
            raw_dataset["validation"],
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=max_seq_length,
        )

    # ── Data Collator ────────────────────────────────────────────────────────
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # ── Training Arguments ───────────────────────────────────────────────────
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        learning_rate=training_config["learning_rate"],
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        weight_decay=training_config.get("weight_decay", 0.01),
        fp16=training_config.get("fp16", True),
        evaluation_strategy=training_config.get("evaluation_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        save_total_limit=training_config.get("save_total_limit", 3),
        logging_steps=training_config.get("logging_steps", 50),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=create_compute_metrics(label_list),
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n🚀 Starting Clinical NER training...")
    print(f"   Model: {config['model']['name']}")
    print(f"   Labels: {len(label_list)}")
    print(f"   Epochs: {training_config['num_train_epochs']}")
    print(f"   Batch size: {training_config['per_device_train_batch_size']}")
    print(f"   Learning rate: {training_config['learning_rate']}")
    print()

    trainer.train()

    # ── Save Final Model ─────────────────────────────────────────────────────
    final_model_path = os.path.join(training_config["output_dir"], "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✅ Model saved to: {final_model_path}")

    # ── Final Evaluation ─────────────────────────────────────────────────────
    print("\n📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    print_metrics_report("Clinical NER (BioBERT)", eval_results)

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Clinical NER Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ner_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()

    train(args.config)
