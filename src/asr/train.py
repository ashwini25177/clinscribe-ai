"""
ClinScribe AI — ASR Training Module
=====================================
Fine-tunes OpenAI Whisper on medical speech data using LoRA (PEFT)
for parameter-efficient training.
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import evaluate
import numpy as np
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.asr.dataset import (
    load_hf_medical_asr_dataset,
    prepare_whisper_dataset,
    WhisperDataCollator,
)
from src.utils.metrics import compute_asr_metrics, print_metrics_report


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model_and_processor(config: dict):
    """
    Initialize Whisper model with LoRA adapters and processor.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (model, processor).
    """
    model_name = config["model"]["name"]

    print(f"📦 Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=config["model"]["language"],
        task=config["model"]["task"],
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Configure model for training
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Required for gradient checkpointing

    # Apply LoRA if enabled
    if config.get("lora", {}).get("enabled", False):
        print("🔧 Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
            bias=config["lora"]["bias"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


def create_compute_metrics(processor):
    """
    Create a metrics computation function for the Trainer.

    Returns a function compatible with HuggingFace Seq2SeqTrainer.
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Compute metrics
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics


def train(config_path: str):
    """
    Main training function for medical ASR.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)

    # ── Setup Model ──────────────────────────────────────────────────────────
    model, processor = setup_model_and_processor(config)

    # ── Load & Prepare Dataset ───────────────────────────────────────────────
    print("📂 Loading medical speech dataset...")

    # Try loading from HuggingFace Hub
    try:
        train_dataset = load_hf_medical_asr_dataset(split="train")
        eval_dataset = load_hf_medical_asr_dataset(split="test")

        train_dataset = prepare_whisper_dataset(train_dataset, processor)
        eval_dataset = prepare_whisper_dataset(eval_dataset, processor)
    except Exception as e:
        print(f"⚠️  Could not load HuggingFace dataset: {e}")
        print("   Falling back to local dataset...")

        from src.asr.dataset import MedicalASRDataset

        train_dataset = MedicalASRDataset(
            data_dir=config["data"]["train_dataset"],
            processor=processor,
            max_audio_length_seconds=config["data"]["max_audio_length_seconds"],
            augment=True,
        )
        eval_dataset = MedicalASRDataset(
            data_dir=config["data"]["eval_dataset"],
            processor=processor,
            max_audio_length_seconds=config["data"]["max_audio_length_seconds"],
            augment=False,
        )

    # ── Data Collator ────────────────────────────────────────────────────────
    data_collator = WhisperDataCollator(processor)

    # ── Training Arguments ───────────────────────────────────────────────────
    training_config = config["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config.get("warmup_steps", 500),
        weight_decay=training_config.get("weight_decay", 0.01),
        fp16=training_config.get("fp16", True),
        evaluation_strategy=training_config.get("evaluation_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 500),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        logging_steps=training_config.get("logging_steps", 100),
        predict_with_generate=True,
        generation_max_length=training_config.get("generation_max_length", 225),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=create_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n🚀 Starting Whisper fine-tuning for medical ASR...")
    print(f"   Model: {config['model']['name']}")
    print(f"   LoRA: {'Enabled' if config.get('lora', {}).get('enabled') else 'Disabled'}")
    print(f"   Epochs: {training_config['num_train_epochs']}")
    print(f"   Batch size: {training_config['per_device_train_batch_size']}")
    print(f"   Learning rate: {training_config['learning_rate']}")
    print()

    trainer.train()

    # ── Save Final Model ─────────────────────────────────────────────────────
    final_model_path = os.path.join(training_config["output_dir"], "final_model")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"\n✅ Model saved to: {final_model_path}")

    # ── Final Evaluation ─────────────────────────────────────────────────────
    print("\n📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    print_metrics_report("Medical ASR (Whisper)", eval_results)

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical ASR Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/asr_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()

    train(args.config)
