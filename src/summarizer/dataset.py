"""
ClinScribe AI — Summarizer Dataset Module
==========================================
Prepares clinical text datasets for abstractive summarization
using Flan-T5 / BART.
"""

import os
import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class ClinicalSummarizationDataset(Dataset):
    """
    PyTorch Dataset for clinical text summarization.

    Expects data in the format:
    [
        {
            "clinical_text": "Patient presents with...",
            "summary": "S: Chief complaint of..."
        },
        ...
    ]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 150,
        prefix: str = "summarize clinical note: ",
    ):
        """
        Args:
            data_path: Path to JSON file with text-summary pairs.
            tokenizer: HuggingFace tokenizer.
            max_input_length: Maximum input sequence length.
            max_target_length: Maximum target (summary) sequence length.
            prefix: Task prefix for T5-style models.
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix

        self.samples = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load summarization data from JSON."""
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Prepare input with task prefix
        input_text = self.prefix + sample["clinical_text"]
        target_text = sample["summary"]

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze()
        # Replace padding token IDs with -100 for loss masking
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_pubmed_summarization_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """
    Load PubMed Summarization dataset from HuggingFace.

    Args:
        split: Dataset split (train, validation, test).
        max_samples: Maximum number of samples to load (for development).
        cache_dir: Cache directory.

    Returns:
        HuggingFace Dataset.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "ccdv/pubmed-summarization",
        "document",
        split=split,
        cache_dir=cache_dir,
    )

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def prepare_hf_summarization_dataset(
    dataset,
    tokenizer,
    max_input_length: int = 512,
    max_target_length: int = 150,
    source_column: str = "article",
    target_column: str = "abstract",
    prefix: str = "summarize clinical note: ",
):
    """
    Prepare a HuggingFace dataset for Seq2Seq summarization training.

    Args:
        dataset: HuggingFace Dataset with source and target columns.
        tokenizer: HuggingFace tokenizer.
        max_input_length: Maximum input length.
        max_target_length: Maximum target length.
        source_column: Source text column name.
        target_column: Target summary column name.
        prefix: Task prefix for T5 models.

    Returns:
        Processed dataset ready for Seq2SeqTrainer.
    """

    def preprocess_function(examples):
        # Add prefix to inputs
        inputs = [prefix + doc for doc in examples[source_column]]
        targets = examples[target_column]

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenize targets
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )

        # Replace pad token IDs with -100
        labels["input_ids"] = [
            [
                (l if l != tokenizer.pad_token_id else -100)
                for l in label
            ]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return processed


def create_clinical_summarization_samples() -> List[Dict]:
    """
    Create sample clinical text-summary pairs for development/testing.

    These are synthetic examples demonstrating the expected data format.
    In production, use actual clinical datasets.

    Returns:
        List of sample data dictionaries.
    """
    samples = [
        {
            "clinical_text": (
                "Patient is a 45-year-old male presenting with persistent cough "
                "for the past two weeks. He reports mild fever, body aches, and "
                "occasional shortness of breath. No history of asthma or COPD. "
                "Vitals: BP 130/85, HR 88, Temp 100.2F, SpO2 96%. Chest X-ray "
                "shows mild bilateral infiltrates. Started on amoxicillin 500mg "
                "three times daily for 7 days. Follow-up in one week."
            ),
            "summary": (
                "S: 45M with 2-week persistent cough, fever, body aches, and "
                "dyspnea. O: BP 130/85, HR 88, T 100.2F, SpO2 96%. CXR: mild "
                "bilateral infiltrates. A: Community-acquired pneumonia. "
                "P: Amoxicillin 500mg TID x 7 days, follow-up 1 week."
            ),
        },
        {
            "clinical_text": (
                "78-year-old female with a history of type 2 diabetes mellitus "
                "and hypertension presents for routine follow-up. Current "
                "medications include metformin 1000mg twice daily and lisinopril "
                "10mg daily. HbA1c is 7.2%, up from 6.8% three months ago. "
                "Blood pressure is 142/88. Patient reports occasional dizziness "
                "and increased thirst. Fasting glucose 145 mg/dL."
            ),
            "summary": (
                "S: 78F with T2DM and HTN for routine follow-up. Reports "
                "dizziness and polydipsia. O: BP 142/88, HbA1c 7.2% (was 6.8%), "
                "FBG 145. A: Suboptimal glycemic and BP control. "
                "P: Increase metformin to 1500mg BID, increase lisinopril to 20mg "
                "daily, recheck in 6 weeks."
            ),
        },
        {
            "clinical_text": (
                "32-year-old female presents with severe headache for 3 days, "
                "associated with photophobia and nausea. No aura reported. "
                "History of episodic migraines since age 20. Takes sumatriptan "
                "50mg as needed, currently averaging 8-10 headache days per month. "
                "Neurological exam is normal. BMI 24. No papilledema on fundoscopy."
            ),
            "summary": (
                "S: 32F with 3-day severe headache with photophobia and nausea. "
                "History of migraines since age 20, 8-10 headache days/month. "
                "O: Neuro exam normal, no papilledema, BMI 24. "
                "A: Chronic migraine without aura, high frequency. "
                "P: Start prophylaxis with topiramate 25mg daily, continue "
                "sumatriptan PRN, headache diary, follow-up 4 weeks."
            ),
        },
    ]

    return samples
