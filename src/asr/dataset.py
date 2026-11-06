"""
ClinScribe AI — ASR Dataset Module
====================================
Handles loading, preprocessing, and preparing medical speech datasets
for Whisper fine-tuning using HuggingFace datasets.
"""

import os
from typing import Dict, Optional

import torch
import numpy as np
from torch.utils.data import Dataset


class MedicalASRDataset(Dataset):
    """
    PyTorch Dataset for medical speech recognition.

    Loads audio files paired with transcriptions and prepares them
    for Whisper fine-tuning.
    """

    def __init__(
        self,
        data_dir: str,
        processor,
        max_audio_length_seconds: int = 30,
        sampling_rate: int = 16000,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing audio files and transcript manifest.
            processor: Whisper processor/feature extractor.
            max_audio_length_seconds: Maximum audio length to keep.
            sampling_rate: Target audio sampling rate.
            augment: Whether to apply audio augmentation.
        """
        self.data_dir = data_dir
        self.processor = processor
        self.max_audio_length = max_audio_length_seconds * sampling_rate
        self.sampling_rate = sampling_rate
        self.augment = augment

        self.samples = self._load_manifest()

    def _load_manifest(self):
        """Load audio-transcription pairs from manifest file."""
        import json

        manifest_path = os.path.join(self.data_dir, "manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback: Try CSV format
        import pandas as pd

        csv_path = os.path.join(self.data_dir, "manifest.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df.to_dict("records")

        raise FileNotFoundError(
            f"No manifest.json or manifest.csv found in {self.data_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        audio_path = os.path.join(self.data_dir, sample["audio_path"])
        transcription = sample["transcription"]

        # Load and preprocess audio
        from src.utils.preprocessing import load_audio, trim_silence, normalize_audio

        audio, sr = load_audio(audio_path, target_sr=self.sampling_rate)
        audio = trim_silence(audio)
        audio = normalize_audio(audio)

        # Truncate if too long
        if len(audio) > self.max_audio_length:
            audio = audio[: self.max_audio_length]

        # Apply augmentation
        if self.augment:
            from src.utils.augmentation import add_noise

            if np.random.random() < 0.3:
                audio = add_noise(audio, noise_level=0.003)

        # Process with Whisper processor
        input_features = self.processor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="np",
        ).input_features[0]

        # Tokenize transcription
        labels = self.processor.tokenizer(transcription).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


def load_hf_medical_asr_dataset(
    dataset_name: str = "ekacare/eka-medical-asr-evaluation-dataset",
    split: str = "train",
    cache_dir: Optional[str] = None,
):
    """
    Load a medical ASR dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split (train, test, validation).
        cache_dir: Local cache directory for downloaded data.

    Returns:
        HuggingFace Dataset object.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
    )
    return dataset


def prepare_whisper_dataset(
    dataset,
    processor,
    max_input_length: int = 30 * 16000,
    audio_column: str = "audio",
    text_column: str = "transcription",
):
    """
    Prepare a HuggingFace dataset for Whisper fine-tuning.

    Applies feature extraction and tokenization in a map function.

    Args:
        dataset: HuggingFace Dataset with audio and text columns.
        processor: WhisperProcessor instance.
        max_input_length: Maximum audio length in samples.
        audio_column: Name of the audio column.
        text_column: Name of the transcription column.

    Returns:
        Processed dataset ready for training.
    """

    def prepare_sample(batch):
        audio = batch[audio_column]

        # Extract input features
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np",
        ).input_features[0]

        # Tokenize labels
        labels = processor.tokenizer(batch[text_column]).input_ids

        batch["input_features"] = input_features
        batch["labels"] = labels

        return batch

    # Filter out audio that is too long
    dataset = dataset.filter(
        lambda x: len(x[audio_column]["array"]) <= max_input_length
    )

    # Apply preprocessing
    dataset = dataset.map(
        prepare_sample,
        remove_columns=dataset.column_names,
    )

    return dataset


class WhisperDataCollator:
    """
    Data collator for Whisper fine-tuning.

    Handles padding of input features and labels, and replaces
    padding token IDs with -100 for loss computation.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Extract input features and labels
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if the decoder adds it
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
