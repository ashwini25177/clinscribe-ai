"""
ClinScribe AI — Summarization Inference Module
================================================
Generate clinical summaries from transcribed medical text
using fine-tuned Flan-T5 / BART.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ClinicalSummarizationInference:
    """
    Clinical text summarization inference engine.

    Generates SOAP-style clinical summaries from transcribed
    doctor-patient conversations or clinical notes.
    """

    def __init__(
        self,
        model_path: str = "checkpoints/summarizer/final_model",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to fine-tuned summarization model.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path) and not ("/" in model_path and not model_path.startswith("checkpoints")):
            fallback = "google/flan-t5-small"
            print(f"⚠️  Summarizer checkpoint not found at '{model_path}'. Using base '{fallback}'...")
            model_path = fallback

        print(f"🔧 Loading summarization model from {model_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print("✅ Summarization model loaded successfully!")

    @torch.no_grad()
    def summarize(
        self,
        clinical_text: str,
        max_length: int = 150,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        prefix: str = "summarize clinical note: ",
    ) -> Dict:
        """
        Generate a clinical summary from input text.

        Args:
            clinical_text: Input clinical text / transcript.
            max_length: Maximum summary length in tokens.
            min_length: Minimum summary length in tokens.
            num_beams: Number of beams for beam search.
            length_penalty: Length penalty (> 1.0 encourages longer outputs).
            no_repeat_ngram_size: Size of n-grams to prevent repetition.
            prefix: Task prefix for T5 models.

        Returns:
            Dictionary with 'summary', 'input_length', and 'processing_time'.
        """
        start_time = time.time()

        # Prepare input
        input_text = prefix + clinical_text
        input_encoding = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate summary
        output_ids = self.model.generate(
            **input_encoding,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )

        summary = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        processing_time = time.time() - start_time

        return {
            "summary": summary,
            "input_length": len(clinical_text.split()),
            "summary_length": len(summary.split()),
            "compression_ratio": round(
                len(summary.split()) / max(len(clinical_text.split()), 1), 3
            ),
            "processing_time_seconds": round(processing_time, 3),
        }

    def summarize_batch(
        self,
        texts: List[str],
        **kwargs,
    ) -> List[Dict]:
        """
        Summarize multiple clinical texts.

        Args:
            texts: List of clinical text strings.
            **kwargs: Additional arguments passed to summarize().

        Returns:
            List of summary result dictionaries.
        """
        results = []
        for i, text in enumerate(texts):
            print(f"  Summarizing [{i + 1}/{len(texts)}]...")
            result = self.summarize(text, **kwargs)
            results.append(result)
        return results

    def generate_soap_note(
        self,
        clinical_text: str,
    ) -> Dict:
        """
        Generate a structured SOAP note from clinical text.

        SOAP format:
        - S (Subjective): Patient's reported symptoms
        - O (Objective): Clinical findings, vitals, test results
        - A (Assessment): Diagnosis / clinical impression
        - P (Plan): Treatment plan, medications, follow-up

        Args:
            clinical_text: Input clinical text.

        Returns:
            Dictionary with SOAP sections and full summary.
        """
        # Generate full summary
        result = self.summarize(
            clinical_text,
            prefix="generate SOAP note from clinical note: ",
            max_length=200,
        )

        summary = result["summary"]

        # Attempt to parse SOAP sections
        soap = {"S": "", "O": "", "A": "", "P": ""}
        current_section = None

        for part in summary.split(". "):
            part = part.strip()
            if part.upper().startswith("S:") or part.upper().startswith("SUBJECTIVE:"):
                current_section = "S"
                part = part.split(":", 1)[1].strip() if ":" in part else part
            elif part.upper().startswith("O:") or part.upper().startswith("OBJECTIVE:"):
                current_section = "O"
                part = part.split(":", 1)[1].strip() if ":" in part else part
            elif part.upper().startswith("A:") or part.upper().startswith("ASSESSMENT:"):
                current_section = "A"
                part = part.split(":", 1)[1].strip() if ":" in part else part
            elif part.upper().startswith("P:") or part.upper().startswith("PLAN:"):
                current_section = "P"
                part = part.split(":", 1)[1].strip() if ":" in part else part

            if current_section and part:
                soap[current_section] += part + ". "

        # Clean up
        soap = {k: v.strip() for k, v in soap.items()}

        # If model generated a narrative summary without explicit S/O/A/P tokens,
        # fallback to providing the narrative summary in Assessment
        if not any(soap.values()) and summary:
            soap["S"] = "Refer to clinical notes / dialogue history"
            soap["A"] = summary
            soap["P"] = "Follow up with patient as prescribed"

        result["soap"] = soap
        return result


def summarize_text(
    text: str,
    model_path: str = "checkpoints/summarizer/final_model",
) -> str:
    """
    Convenience function to summarize clinical text.

    Args:
        text: Clinical text.
        model_path: Path to fine-tuned model.

    Returns:
        Summary string.
    """
    engine = ClinicalSummarizationInference(model_path=model_path)
    result = engine.summarize(text)
    return result["summary"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clinical Summarization Inference")
    parser.add_argument("--text", type=str, required=True, help="Clinical text to summarize")
    parser.add_argument(
        "--model", type=str, default="checkpoints/summarizer/final_model",
        help="Path to fine-tuned model",
    )
    parser.add_argument("--soap", action="store_true", help="Generate SOAP note format")
    args = parser.parse_args()

    engine = ClinicalSummarizationInference(model_path=args.model)

    if args.soap:
        result = engine.generate_soap_note(args.text)
        print(f"\n📋 SOAP Note:")
        for section, content in result["soap"].items():
            print(f"  {section}: {content}")
    else:
        result = engine.summarize(args.text)
        print(f"\n📝 Summary:\n{result['summary']}")

    print(f"\n📊 Compression: {result['input_length']} → {result['summary_length']} words")
    print(f"⚡ Processing time: {result['processing_time_seconds']}s")
