"""
ClinScribe AI — Unified Metrics Module
=======================================
Provides consistent metric computation across all modules:
  - ASR: WER, CER
  - NER: Precision, Recall, F1 (entity-level via seqeval)
  - Summarization: ROUGE-1/2/L, BLEU
"""

import numpy as np
from typing import Dict, List, Any


# =============================================================================
# ASR Metrics
# =============================================================================

def compute_asr_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER).

    Args:
        predictions: List of predicted transcription strings.
        references: List of ground truth transcription strings.

    Returns:
        Dictionary with 'wer' and 'cer' scores.
    """
    from jiwer import wer, cer

    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)

    return {
        "wer": round(word_error_rate, 4),
        "cer": round(char_error_rate, 4),
    }


def compute_medical_keyword_accuracy(
    predictions: List[str],
    references: List[str],
    medical_terms: List[str],
) -> Dict[str, float]:
    """
    Compute accuracy specifically for medical keywords in transcriptions.
    Measures how well the ASR model captures domain-specific terms.

    Args:
        predictions: Predicted transcripts.
        references: Reference transcripts.
        medical_terms: List of medical terms to track.

    Returns:
        Dictionary with 'medical_keyword_accuracy' and per-term breakdown.
    """
    medical_terms_lower = [term.lower() for term in medical_terms]
    total_expected = 0
    total_found = 0
    per_term_stats = {}

    for term in medical_terms_lower:
        expected = sum(1 for ref in references if term in ref.lower())
        found = sum(
            1
            for pred, ref in zip(predictions, references)
            if term in ref.lower() and term in pred.lower()
        )
        total_expected += expected
        total_found += found
        if expected > 0:
            per_term_stats[term] = round(found / expected, 4)

    overall_accuracy = total_found / total_expected if total_expected > 0 else 0.0

    return {
        "medical_keyword_accuracy": round(overall_accuracy, 4),
        "per_term_accuracy": per_term_stats,
        "total_expected": total_expected,
        "total_found": total_found,
    }


# =============================================================================
# NER Metrics
# =============================================================================

def compute_ner_metrics(
    predictions: List[List[str]],
    references: List[List[str]],
    mode: str = "strict",
) -> Dict[str, float]:
    """
    Compute entity-level Precision, Recall, and F1 using seqeval.

    Args:
        predictions: List of predicted label sequences (BIO format).
        references: List of true label sequences (BIO format).
        mode: Evaluation mode — 'strict' (exact match) or 'lenient'.

    Returns:
        Dictionary with precision, recall, f1, and accuracy.
    """
    from seqeval.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        classification_report,
    )

    precision = precision_score(references, predictions, mode=mode)
    recall = recall_score(references, predictions, mode=mode)
    f1 = f1_score(references, predictions, mode=mode)
    accuracy = accuracy_score(references, predictions)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "classification_report": classification_report(references, predictions),
    }


def compute_ner_metrics_per_entity(
    predictions: List[List[str]],
    references: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-entity-type metrics for detailed analysis.

    Returns:
        Nested dictionary: {entity_type: {precision, recall, f1, support}}.
    """
    from seqeval.metrics import classification_report
    import re

    report = classification_report(references, predictions, output_dict=True)
    # Filter out aggregate keys
    entity_metrics = {
        k: v for k, v in report.items()
        if k not in ("micro avg", "macro avg", "weighted avg", "samples avg")
    }
    return entity_metrics


# =============================================================================
# Summarization Metrics
# =============================================================================

def compute_summarization_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE and BLEU scores for summarization evaluation.

    Args:
        predictions: List of generated summary strings.
        references: List of reference summary strings.

    Returns:
        Dictionary with rouge1, rouge2, rougeL, and bleu scores.
    """
    # ROUGE via rouge_score (offline, immediate)
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for ref, pred in zip(references, predictions):
            scores = scorer.score(ref, pred)
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rl.append(scores["rougeL"].fmeasure)
        rouge_results = {
            "rouge1": np.mean(r1) if r1 else 0.0,
            "rouge2": np.mean(r2) if r2 else 0.0,
            "rougeL": np.mean(rl) if rl else 0.0,
        }
    except Exception:
        import evaluate
        rouge_metric = evaluate.load("rouge")
        rouge_results = rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )

    # BLEU
    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])

    return {
        "rouge1": round(rouge_results["rouge1"], 4),
        "rouge2": round(rouge_results["rouge2"], 4),
        "rougeL": round(rouge_results["rougeL"], 4),
        "bleu": round(bleu_score.score, 4),
    }


# =============================================================================
# Utility: Print Metrics Report
# =============================================================================

def print_metrics_report(
    module_name: str,
    metrics: Dict[str, Any],
) -> None:
    """Pretty-print a metrics report to console."""
    print(f"\n{'=' * 60}")
    print(f"  📊 {module_name} — Evaluation Results")
    print(f"{'=' * 60}")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        elif isinstance(value, str):
            print(f"\n  {key}:\n{value}")
        else:
            print(f"  {key}: {value}")
    print(f"{'=' * 60}\n")
