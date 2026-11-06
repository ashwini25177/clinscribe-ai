"""
ClinScribe AI — NER Dataset Module
====================================
Prepares clinical text datasets for token classification (NER)
using BioBERT / ClinicalBERT with BIO tagging scheme.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# =============================================================================
# BIO Tag Utilities
# =============================================================================

# Default clinical entity labels
DEFAULT_LABEL_LIST = [
    "O",
    "B-DRUG", "I-DRUG",
    "B-DOSAGE", "I-DOSAGE",
    "B-FORM", "I-FORM",
    "B-ROUTE", "I-ROUTE",
    "B-FREQUENCY", "I-FREQUENCY",
    "B-DURATION", "I-DURATION",
    "B-CONDITION", "I-CONDITION",
]


def get_label_maps(label_list: List[str] = None) -> Tuple[Dict, Dict]:
    """
    Create label-to-id and id-to-label mappings.

    Args:
        label_list: List of BIO labels. Uses DEFAULT_LABEL_LIST if None.

    Returns:
        Tuple of (label2id, id2label) dictionaries.
    """
    if label_list is None:
        label_list = DEFAULT_LABEL_LIST

    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}
    return label2id, id2label


# =============================================================================
# NER Dataset
# =============================================================================

class ClinicalNERDataset(Dataset):
    """
    PyTorch Dataset for clinical Named Entity Recognition.

    Expects data in the format:
    [
        {
            "tokens": ["Patient", "takes", "metformin", "500mg", "daily"],
            "ner_tags": ["O", "O", "B-DRUG", "B-DOSAGE", "B-FREQUENCY"]
        },
        ...
    ]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        label_list: List[str] = None,
        max_seq_length: int = 512,
    ):
        """
        Args:
            data_path: Path to JSON file with token-label pairs.
            tokenizer: HuggingFace tokenizer.
            label_list: List of BIO entity labels.
            max_seq_length: Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if label_list is None:
            label_list = DEFAULT_LABEL_LIST
        self.label2id, self.id2label = get_label_maps(label_list)

        self.samples = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load NER data from JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # Tokenize with subword alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = self._align_labels(ner_tags, word_ids)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }

    def _align_labels(
        self,
        labels: List[str],
        word_ids: List[Optional[int]],
    ) -> List[int]:
        """
        Align BIO labels with subword tokens.

        For subword tokens that belong to the same original word:
        - First subword gets the original label
        - Subsequent subwords get the I- version of the label
        - Special tokens ([CLS], [SEP], [PAD]) get label -100

        Args:
            labels: Original BIO labels for each word.
            word_ids: Word indices from the tokenizer.

        Returns:
            List of aligned label IDs.
        """
        aligned = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens
                aligned.append(-100)
            elif word_id != previous_word_id:
                # First subword of a new word
                label = labels[word_id] if word_id < len(labels) else "O"
                aligned.append(self.label2id.get(label, 0))
            else:
                # Subsequent subword of the same word
                label = labels[word_id] if word_id < len(labels) else "O"
                # Convert B- to I- for subwords
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                aligned.append(self.label2id.get(label, 0))

            previous_word_id = word_id

        return aligned


# =============================================================================
# Dataset Loading from HuggingFace
# =============================================================================

def load_bc5cdr_dataset(
    split: str = "train",
    cache_dir: Optional[str] = None,
):
    """
    Load BC5CDR dataset from HuggingFace for chemical/disease NER.

    Args:
        split: Dataset split.
        cache_dir: Cache directory.

    Returns:
        HuggingFace Dataset.
    """
    from datasets import load_dataset

    dataset = load_dataset("bigbio/bc5cdr", split=split, cache_dir=cache_dir)
    return dataset


def load_ncbi_disease_dataset(
    split: str = "train",
    cache_dir: Optional[str] = None,
):
    """
    Load NCBI Disease corpus from HuggingFace.

    Args:
        split: Dataset split.
        cache_dir: Cache directory.

    Returns:
        HuggingFace Dataset.
    """
    from datasets import load_dataset

    dataset = load_dataset("ncbi_disease", split=split, cache_dir=cache_dir)
    return dataset


def prepare_hf_ner_dataset(
    dataset,
    tokenizer,
    label_list: List[str],
    max_seq_length: int = 512,
    tokens_column: str = "tokens",
    tags_column: str = "ner_tags",
):
    """
    Prepare a HuggingFace NER dataset for training.

    Handles subword alignment and creates proper label tensors.

    Args:
        dataset: HuggingFace Dataset with tokens and tags.
        tokenizer: HuggingFace tokenizer.
        label_list: List of BIO labels.
        max_seq_length: Maximum sequence length.
        tokens_column: Column name for tokens.
        tags_column: Column name for NER tags.

    Returns:
        Processed dataset ready for training.
    """
    label2id = {label: idx for idx, label in enumerate(label_list)}

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples[tokens_column],
            is_split_into_words=True,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        all_labels = []
        for i, labels in enumerate(examples[tags_column]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)
                elif word_id != previous_word_id:
                    aligned.append(labels[word_id])
                else:
                    label = labels[word_id]
                    # If using string labels, convert B- to I-
                    if isinstance(label, str) and label.startswith("B-"):
                        label = label2id.get("I-" + label[2:], label2id.get(label, 0))
                    aligned.append(label)

                previous_word_id = word_id

            all_labels.append(aligned)

        tokenized["labels"] = all_labels
        return tokenized

    processed = dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return processed


# =============================================================================
# MTSamples Processing
# =============================================================================

def load_mtsamples_for_ner(csv_path: str) -> List[Dict]:
    """
    Load and prepare MTSamples dataset for NER annotation.

    Since MTSamples is unlabeled, this function prepares the raw text
    for annotation with scispaCy or manual labeling.

    Args:
        csv_path: Path to mtsamples.csv file.

    Returns:
        List of dictionaries with tokenized text ready for annotation.
    """
    import pandas as pd
    from src.utils.preprocessing import clean_clinical_text, tokenize_for_ner

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["transcription"])

    samples = []
    for _, row in df.iterrows():
        text = clean_clinical_text(row["transcription"])
        tokens = tokenize_for_ner(text)

        samples.append({
            "specialty": row.get("medical_specialty", "Unknown"),
            "tokens": tokens,
            "text": text,
            "ner_tags": ["O"] * len(tokens),  # Placeholder — needs annotation
        })

    return samples
