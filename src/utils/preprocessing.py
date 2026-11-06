"""
ClinScribe AI — Preprocessing Utilities
========================================
Shared preprocessing functions for audio and text data across all modules.
"""

import re
import os
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Audio Preprocessing
# =============================================================================

def load_audio(
    audio_path: str,
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        audio_path: Path to audio file (.wav, .mp3, .flac).
        target_sr: Target sampling rate (Whisper expects 16kHz).

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    import librosa

    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr


def trim_silence(
    audio: np.ndarray,
    top_db: int = 30,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio: Audio waveform array.
        top_db: Threshold in dB below reference to consider as silence.

    Returns:
        Trimmed audio array.
    """
    import librosa

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def chunk_audio(
    audio: np.ndarray,
    sr: int = 16000,
    chunk_duration_seconds: int = 30,
    overlap_seconds: int = 2,
) -> List[np.ndarray]:
    """
    Split long audio into overlapping chunks for processing.

    Args:
        audio: Audio waveform array.
        sr: Sample rate.
        chunk_duration_seconds: Duration of each chunk.
        overlap_seconds: Overlap between consecutive chunks.

    Returns:
        List of audio chunks.
    """
    chunk_size = chunk_duration_seconds * sr
    overlap_size = overlap_seconds * sr
    step_size = chunk_size - overlap_size

    chunks = []
    for start in range(0, len(audio), step_size):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        if len(chunk) > sr:  # Skip chunks shorter than 1 second
            chunks.append(chunk)

    return chunks


# =============================================================================
# Text Preprocessing
# =============================================================================

def clean_clinical_text(text: str) -> str:
    """
    Clean and normalize clinical text for NLP processing.

    Performs:
    - Lowercasing
    - Whitespace normalization
    - Removal of excessive punctuation
    - Preserves medical abbreviations and dosages

    Args:
        text: Raw clinical text.

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", "", text)

    # Normalize dashes and hyphens
    text = re.sub(r"[–—]", "-", text)

    # Remove repeated punctuation (but keep single instances)
    text = re.sub(r"([.!?]){2,}", r"\1", text)

    return text.strip()


def normalize_medical_text(text: str) -> str:
    """
    Normalize medical-specific text patterns.

    Handles:
    - Dosage patterns (10 mg → 10mg)
    - Common abbreviation standardization
    - Unit normalization

    Args:
        text: Clinical text string.

    Returns:
        Normalized text.
    """
    # Normalize dosage patterns: "10 mg" → "10mg", "500 mL" → "500mL"
    text = re.sub(
        r"(\d+)\s*(mg|mcg|ml|mL|g|kg|units?|tabs?|caps?)\b",
        r"\1\2",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize common abbreviations
    abbreviation_map = {
        r"\bb\.i\.d\.?\b": "BID",          # Twice daily
        r"\bt\.i\.d\.?\b": "TID",          # Three times daily
        r"\bq\.i\.d\.?\b": "QID",          # Four times daily
        r"\bq\.d\.?\b": "QD",              # Once daily
        r"\bp\.r\.n\.?\b": "PRN",          # As needed
        r"\bp\.o\.?\b": "PO",              # By mouth
        r"\bI\.V\.?\b": "IV",              # Intravenous
        r"\bI\.M\.?\b": "IM",              # Intramuscular
        r"\bS\.C\.?\b": "SC",              # Subcutaneous
    }

    for pattern, replacement in abbreviation_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def tokenize_for_ner(
    text: str,
    preserve_case: bool = True,
) -> List[str]:
    """
    Tokenize clinical text for NER, preserving medical terms.

    Uses whitespace + punctuation tokenization that preserves:
    - Hyphenated terms (e.g., "beta-blocker")
    - Dosage patterns (e.g., "500mg")
    - Decimal numbers (e.g., "3.5")

    Args:
        text: Text to tokenize.
        preserve_case: If True, preserves original casing.

    Returns:
        List of tokens.
    """
    # Split on whitespace but keep punctuation attached to medical terms
    tokens = re.findall(
        r"\b[\w][\w\-./]*[\w]\b|\b\w\b|[^\s\w]",
        text,
    )

    if not preserve_case:
        tokens = [t.lower() for t in tokens]

    return tokens


# =============================================================================
# Dataset Utilities
# =============================================================================

def create_train_val_split(
    data: list,
    val_ratio: float = 0.15,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: List of data samples.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))

    test_size = int(len(data) * test_ratio)
    val_size = int(len(data) * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path
