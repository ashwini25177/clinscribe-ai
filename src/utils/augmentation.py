"""
ClinScribe AI — Data Augmentation Utilities
============================================
Augmentation strategies for both audio and text data to improve
model robustness and generalization.
"""

import random
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Audio Augmentation
# =============================================================================

class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.

    Applies frequency masking and time masking to mel spectrograms
    to improve ASR model robustness.

    Reference: Park et al., 2019 (https://arxiv.org/abs/1904.08779)
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to a mel spectrogram.

        Args:
            mel_spectrogram: 2D array of shape (n_mels, time_steps).

        Returns:
            Augmented mel spectrogram.
        """
        augmented = mel_spectrogram.copy()
        n_mels, time_steps = augmented.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels))
            f0 = random.randint(0, n_mels - f)
            augmented[f0 : f0 + f, :] = 0

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_steps))
            t0 = random.randint(0, time_steps - t)
            augmented[:, t0 : t0 + t] = 0

        return augmented


def add_noise(
    audio: np.ndarray,
    noise_level: float = 0.005,
) -> np.ndarray:
    """
    Add Gaussian noise to audio waveform.

    Args:
        audio: Audio waveform array.
        noise_level: Standard deviation of the noise.

    Returns:
        Noisy audio array.
    """
    noise = np.random.randn(*audio.shape) * noise_level
    return audio + noise


def speed_perturbation(
    audio: np.ndarray,
    sr: int = 16000,
    speed_factor: float = 1.0,
) -> np.ndarray:
    """
    Apply speed perturbation to audio.

    Args:
        audio: Audio waveform array.
        sr: Sample rate.
        speed_factor: Speed multiplier (< 1.0 = slower, > 1.0 = faster).

    Returns:
        Speed-perturbed audio.
    """
    import librosa

    return librosa.effects.time_stretch(audio, rate=speed_factor)


def pitch_shift(
    audio: np.ndarray,
    sr: int = 16000,
    n_steps: float = 0.0,
) -> np.ndarray:
    """
    Apply pitch shifting to audio.

    Args:
        audio: Audio waveform array.
        sr: Sample rate.
        n_steps: Number of semitones to shift (positive = up, negative = down).

    Returns:
        Pitch-shifted audio.
    """
    import librosa

    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


# =============================================================================
# Text Augmentation
# =============================================================================

# Medical synonym dictionary for entity-preserving augmentation
MEDICAL_SYNONYMS = {
    # Conditions
    "hypertension": ["high blood pressure", "HTN", "elevated blood pressure"],
    "diabetes mellitus": ["diabetes", "DM", "sugar disease"],
    "myocardial infarction": ["heart attack", "MI", "cardiac infarction"],
    "pneumonia": ["lung infection", "pulmonary infection"],
    "bronchitis": ["airway inflammation", "bronchial infection"],
    "migraine": ["severe headache", "vascular headache"],
    "arthritis": ["joint inflammation", "joint disease"],
    "anemia": ["low hemoglobin", "low blood count"],
    # Routes
    "oral": ["by mouth", "PO", "orally"],
    "intravenous": ["IV", "intravenously"],
    "intramuscular": ["IM"],
    "subcutaneous": ["SC", "subQ"],
    "topical": ["topically", "applied to the skin"],
    # Frequencies
    "twice daily": ["BID", "two times a day", "b.i.d."],
    "three times daily": ["TID", "three times a day", "t.i.d."],
    "once daily": ["QD", "daily", "once a day", "q.d."],
    "as needed": ["PRN", "when needed", "p.r.n."],
}


def synonym_replacement(
    text: str,
    n_replacements: int = 1,
    synonym_dict: Optional[dict] = None,
) -> str:
    """
    Replace medical terms with their synonyms for data augmentation.

    Args:
        text: Input clinical text.
        n_replacements: Number of terms to replace.
        synonym_dict: Custom synonym dictionary. Uses MEDICAL_SYNONYMS by default.

    Returns:
        Augmented text with synonym replacements.
    """
    if synonym_dict is None:
        synonym_dict = MEDICAL_SYNONYMS

    text_lower = text.lower()
    replacements_made = 0
    result = text

    # Find which terms exist in the text
    found_terms = [
        term for term in synonym_dict if term.lower() in text_lower
    ]

    if not found_terms:
        return result

    random.shuffle(found_terms)

    for term in found_terms:
        if replacements_made >= n_replacements:
            break

        synonyms = synonym_dict[term]
        replacement = random.choice(synonyms)

        # Case-preserving replacement
        import re

        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(replacement, result, count=1)
        replacements_made += 1

    return result


def random_deletion(
    tokens: List[str],
    p: float = 0.1,
    protected_labels: Optional[List[str]] = None,
) -> List[str]:
    """
    Randomly delete non-entity tokens with probability p.

    Args:
        tokens: List of word tokens.
        p: Probability of deleting each token.
        protected_labels: Entity label indices to preserve (not delete).

    Returns:
        List of tokens with some randomly removed.
    """
    if len(tokens) <= 1:
        return tokens

    result = []
    for i, token in enumerate(tokens):
        if random.random() > p:
            result.append(token)

    # Ensure at least one token remains
    if not result:
        result = [random.choice(tokens)]

    return result


def random_swap(tokens: List[str], n_swaps: int = 1) -> List[str]:
    """
    Randomly swap positions of n token pairs.

    Args:
        tokens: List of word tokens.
        n_swaps: Number of swap operations.

    Returns:
        Tokens with random swaps applied.
    """
    result = tokens.copy()

    for _ in range(n_swaps):
        if len(result) < 2:
            break
        idx1, idx2 = random.sample(range(len(result)), 2)
        result[idx1], result[idx2] = result[idx2], result[idx1]

    return result
