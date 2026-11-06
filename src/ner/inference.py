"""
ClinScribe AI — NER Inference Module
======================================
Extract medical entities from clinical text using the fine-tuned BioBERT model.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ClinicalNERInference:
    """
    Clinical Named Entity Recognition inference engine.

    Extracts medical entities (drugs, dosages, conditions, etc.)
    from clinical text using fine-tuned BioBERT.
    """

    def __init__(
        self,
        model_path: str = "checkpoints/ner/final_model",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to the fine-tuned NER model.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path) and not ("/" in model_path and not model_path.startswith("checkpoints")):
            fallback = "d4data/biomedical-ner-all"
            print(f"⚠️  NER checkpoint not found at '{model_path}'. Using pretrained '{fallback}'...")
            model_path = fallback

        print(f"🔧 Loading NER model from {model_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Create NER pipeline for convenient usage
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple",  # Merge subword tokens
        )

        print("✅ NER model loaded successfully!")

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract medical entities from clinical text.

        Args:
            text: Clinical text string.

        Returns:
            List of entity dictionaries with keys:
            - text: Entity text span
            - label: Entity type (DRUG, DOSAGE, CONDITION, etc.)
            - start: Start character offset
            - end: End character offset
            - score: Confidence score
        """
        if not text or not text.strip():
            return []

        raw_entities = self.ner_pipeline(text)

        entities = []
        for entity in raw_entities:
            entities.append({
                "text": entity["word"].strip(),
                "label": entity["entity_group"],
                "start": entity["start"],
                "end": entity["end"],
                "score": round(float(entity["score"]), 4),
            })

        # Post-process: merge adjacent entities of the same type
        entities = self._merge_adjacent_entities(entities)

        return entities

    def extract_entities_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of clinical text strings.

        Returns:
            List of entity lists for each text.
        """
        return [self.extract_entities(text) for text in texts]

    def extract_structured(self, text: str) -> Dict:
        """
        Extract entities and organize them by category.

        Args:
            text: Clinical text string.

        Returns:
            Dictionary organized by entity type:
            {
                "drugs": [...],
                "dosages": [...],
                "conditions": [...],
                ...
            }
        """
        entities = self.extract_entities(text)

        structured = {
            "drugs": [],
            "dosages": [],
            "forms": [],
            "routes": [],
            "frequencies": [],
            "durations": [],
            "conditions": [],
        }

        label_to_key = {
            "DRUG": "drugs",
            "DOSAGE": "dosages",
            "FORM": "forms",
            "ROUTE": "routes",
            "FREQUENCY": "frequencies",
            "DURATION": "durations",
            "CONDITION": "conditions",
        }

        for entity in entities:
            key = label_to_key.get(entity["label"])
            if key:
                structured[key].append({
                    "text": entity["text"],
                    "score": entity["score"],
                    "start": entity["start"],
                    "end": entity["end"],
                })

        return structured

    @staticmethod
    def _merge_adjacent_entities(entities: List[Dict]) -> List[Dict]:
        """
        Merge adjacent entities of the same type.

        Example: ["aspir", "in"] with same label → "aspirin"
        """
        if not entities:
            return entities

        merged = [entities[0].copy()]

        for entity in entities[1:]:
            prev = merged[-1]
            # Check if same type and adjacent (within 1 character)
            if (
                entity["label"] == prev["label"]
                and entity["start"] - prev["end"] <= 1
            ):
                # Merge
                prev["text"] = prev["text"] + entity["text"].lstrip("#")
                prev["end"] = entity["end"]
                prev["score"] = min(prev["score"], entity["score"])
            else:
                merged.append(entity.copy())

        return merged


def extract_entities(
    text: str,
    model_path: str = "checkpoints/ner/final_model",
) -> List[Dict]:
    """
    Convenience function to extract entities from clinical text.

    Args:
        text: Clinical text.
        model_path: Path to fine-tuned model.

    Returns:
        List of entity dictionaries.
    """
    engine = ClinicalNERInference(model_path=model_path)
    return engine.extract_entities(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clinical NER Inference")
    parser.add_argument("--text", type=str, required=True, help="Clinical text to analyze")
    parser.add_argument(
        "--model", type=str, default="checkpoints/ner/final_model",
        help="Path to fine-tuned model",
    )
    args = parser.parse_args()

    engine = ClinicalNERInference(model_path=args.model)
    entities = engine.extract_entities(args.text)

    print(f"\n📋 Input: {args.text}")
    print(f"\n🏷️  Extracted Entities ({len(entities)}):")
    for entity in entities:
        print(
            f"  [{entity['label']}] \"{entity['text']}\" "
            f"(score: {entity['score']}, pos: {entity['start']}-{entity['end']})"
        )
