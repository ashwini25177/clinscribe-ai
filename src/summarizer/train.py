"""
ClinScribe AI — Summarization Training Module
===============================================
Fine-tunes Flan-T5 / BART for clinical note summarization
using HuggingFace Seq2SeqTrainer.
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.summarizer.dataset import (
    ClinicalSummarizationDataset,
    load_pubmed_summarization_dataset,
    prepare_hf_summarization_dataset,
)
from src.utils.metrics import compute_summarization_metrics, print_metrics_report


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: dict):
    """
    Initialize Flan-T5 / BART model and tokenizer.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (model, tokenizer).
    """
    model_name = config["model"]["name"]

    print(f"📦 Loading summarization model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def create_compute_metrics(tokenizer):
    """
    Create metrics function for Seq2SeqTrainer.

    Computes ROUGE and BLEU scores.
    """
    rouge_metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Replace -100 with pad token for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # Strip whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute ROUGE
        rouge_results = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        return {
            "rouge1": round(rouge_results["rouge1"], 4),
            "rouge2": round(rouge_results["rouge2"], 4),
            "rouge_l": round(rouge_results["rougeL"], 4),
        }

    return compute_metrics


def train(config_path: str):
    """
    Main training function for clinical summarization.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)

    # ── Setup Model ──────────────────────────────────────────────────────────
    model, tokenizer = setup_model_and_tokenizer(config)

    # ── Load Dataset ─────────────────────────────────────────────────────────
    print("📂 Loading summarization dataset...")

    data_config = config["data"]
    max_input_length = data_config.get("max_input_length", 512)
    max_target_length = data_config.get("max_target_length", 150)
    prefix = data_config.get("prefix", "summarize clinical note: ")

    # Try loading from local files first
    train_path = data_config.get("train_dataset")

    if train_path and os.path.exists(train_path):
        train_dataset = ClinicalSummarizationDataset(
            data_path=os.path.join(train_path, "data.json"),
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            prefix=prefix,
        )
        eval_dataset = ClinicalSummarizationDataset(
            data_path=os.path.join(data_config["eval_dataset"], "data.json"),
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            prefix=prefix,
        )
    else:
        # Fallback: Load PubMed summarization from HuggingFace
        print("   Loading PubMed Summarization from HuggingFace...")

        raw_train = load_pubmed_summarization_dataset(
            split="train", max_samples=5000  # Limit for faster iteration
        )
        raw_eval = load_pubmed_summarization_dataset(
            split="validation", max_samples=500
        )

        train_dataset = prepare_hf_summarization_dataset(
            raw_train,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            prefix=prefix,
        )
        eval_dataset = prepare_hf_summarization_dataset(
            raw_eval,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            prefix=prefix,
        )

    # ── Data Collator ────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # ── Training Arguments ───────────────────────────────────────────────────
    training_config = config["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config.get("warmup_steps", 200),
        weight_decay=training_config.get("weight_decay", 0.01),
        fp16=training_config.get("fp16", True),
        evaluation_strategy=training_config.get("evaluation_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        save_total_limit=training_config.get("save_total_limit", 3),
        logging_steps=training_config.get("logging_steps", 100),
        predict_with_generate=True,
        generation_max_length=training_config.get("generation_max_length", 150),
        generation_num_beams=training_config.get("generation_num_beams", 4),
        load_best_model_at_end=True,
        metric_for_best_model="rouge_l",
        greater_is_better=True,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=create_compute_metrics(tokenizer),
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n🚀 Starting Clinical Summarization training...")
    print(f"   Model: {config['model']['name']}")
    print(f"   Epochs: {training_config['num_train_epochs']}")
    print(f"   Batch size: {training_config['per_device_train_batch_size']}")
    print(f"   Grad accum: {training_config.get('gradient_accumulation_steps', 4)}")
    print(f"   Learning rate: {training_config['learning_rate']}")
    print(f"   Beam search: {training_config.get('generation_num_beams', 4)} beams")
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
    print_metrics_report("Clinical Summarization (Flan-T5)", eval_results)

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Clinical Summarizer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/summarizer_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()

    train(args.config)
