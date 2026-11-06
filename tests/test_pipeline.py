"""
ClinScribe AI — Unit Tests
============================
Tests for preprocessing, augmentation, metrics, and dataset utilities.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Preprocessing Tests
# =============================================================================

class TestTextPreprocessing:
    """Tests for clinical text preprocessing utilities."""

    def test_clean_clinical_text_whitespace(self):
        from src.utils.preprocessing import clean_clinical_text

        text = "Patient  has   a  history  of   diabetes."
        result = clean_clinical_text(text)
        assert "  " not in result
        assert result == "Patient has a history of diabetes."

    def test_clean_clinical_text_empty(self):
        from src.utils.preprocessing import clean_clinical_text

        assert clean_clinical_text("") == ""
        assert clean_clinical_text(None) == ""

    def test_normalize_medical_text_dosages(self):
        from src.utils.preprocessing import normalize_medical_text

        text = "Take 500 mg of amoxicillin"
        result = normalize_medical_text(text)
        assert "500mg" in result

    def test_normalize_medical_text_abbreviations(self):
        from src.utils.preprocessing import normalize_medical_text

        text = "Take medication b.i.d."
        result = normalize_medical_text(text)
        assert "BID" in result

    def test_tokenize_for_ner(self):
        from src.utils.preprocessing import tokenize_for_ner

        text = "Patient takes metformin 500mg daily"
        tokens = tokenize_for_ner(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "Patient" in tokens

    def test_train_val_split(self):
        from src.utils.preprocessing import create_train_val_split

        data = list(range(100))
        train, val, test = create_train_val_split(data, val_ratio=0.15, test_ratio=0.1)
        assert len(train) + len(val) + len(test) == 100
        assert len(test) == 10
        assert len(val) == 15


# =============================================================================
# Augmentation Tests
# =============================================================================

class TestAugmentation:
    """Tests for data augmentation utilities."""

    def test_spec_augment(self):
        from src.utils.augmentation import SpecAugment

        spec_aug = SpecAugment(
            freq_mask_param=5, time_mask_param=10,
            num_freq_masks=1, num_time_masks=1,
        )
        mel = np.random.randn(80, 100)
        augmented = spec_aug(mel)
        assert augmented.shape == mel.shape
        # At least some values should be zeroed
        assert np.sum(augmented == 0) > np.sum(mel == 0)

    def test_add_noise(self):
        from src.utils.augmentation import add_noise

        audio = np.zeros(16000)
        noisy = add_noise(audio, noise_level=0.01)
        assert noisy.shape == audio.shape
        assert not np.allclose(audio, noisy)

    def test_synonym_replacement(self):
        from src.utils.augmentation import synonym_replacement

        text = "Patient has hypertension and diabetes mellitus"
        result = synonym_replacement(text, n_replacements=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_random_deletion(self):
        from src.utils.augmentation import random_deletion

        tokens = ["The", "patient", "has", "fever", "and", "cough"]
        result = random_deletion(tokens, p=0.5)
        assert len(result) >= 1  # At least one token preserved
        assert len(result) <= len(tokens)

    def test_random_swap(self):
        from src.utils.augmentation import random_swap

        tokens = ["one", "two", "three", "four", "five"]
        result = random_swap(tokens, n_swaps=1)
        assert len(result) == len(tokens)
        assert set(result) == set(tokens)  # Same elements, potentially different order


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Tests for metric computation utilities."""

    def test_asr_metrics(self):
        from src.utils.metrics import compute_asr_metrics

        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        metrics = compute_asr_metrics(predictions, references)
        assert metrics["wer"] == 0.0
        assert metrics["cer"] == 0.0

    def test_asr_metrics_with_errors(self):
        from src.utils.metrics import compute_asr_metrics

        predictions = ["the cat sat on a mat"]
        references = ["the cat sat on the mat"]
        metrics = compute_asr_metrics(predictions, references)
        assert metrics["wer"] > 0.0

    def test_ner_metrics(self):
        from src.utils.metrics import compute_ner_metrics

        predictions = [["O", "B-DRUG", "O", "B-CONDITION"]]
        references = [["O", "B-DRUG", "O", "B-CONDITION"]]
        metrics = compute_ner_metrics(predictions, references)
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_medical_keyword_accuracy(self):
        from src.utils.metrics import compute_medical_keyword_accuracy

        predictions = ["patient has hypertension and diabetes"]
        references = ["patient has hypertension and diabetes"]
        terms = ["hypertension", "diabetes"]
        metrics = compute_medical_keyword_accuracy(predictions, references, terms)
        assert metrics["medical_keyword_accuracy"] == 1.0

    def test_print_metrics_report(self):
        from src.utils.metrics import print_metrics_report

        metrics = {"wer": 0.15, "cer": 0.08}
        # Should not raise
        print_metrics_report("Test Module", metrics)


# =============================================================================
# NER Dataset Tests
# =============================================================================

class TestNERDataset:
    """Tests for NER dataset utilities."""

    def test_label_maps(self):
        from src.ner.dataset import get_label_maps, DEFAULT_LABEL_LIST

        label2id, id2label = get_label_maps()
        assert label2id["O"] == 0
        assert id2label[0] == "O"
        assert len(label2id) == len(DEFAULT_LABEL_LIST)
        assert len(id2label) == len(DEFAULT_LABEL_LIST)

    def test_custom_label_maps(self):
        from src.ner.dataset import get_label_maps

        custom = ["O", "B-DISEASE", "I-DISEASE"]
        label2id, id2label = get_label_maps(custom)
        assert len(label2id) == 3
        assert label2id["B-DISEASE"] == 1


# =============================================================================
# Summarizer Dataset Tests
# =============================================================================

class TestSummarizerDataset:
    """Tests for summarization dataset utilities."""

    def test_create_sample_data(self):
        from src.summarizer.dataset import create_clinical_summarization_samples

        samples = create_clinical_summarization_samples()
        assert len(samples) >= 3
        for sample in samples:
            assert "clinical_text" in sample
            assert "summary" in sample
            assert len(sample["clinical_text"]) > 50
            assert len(sample["summary"]) > 20


# =============================================================================
# API Schema Tests
# =============================================================================

class TestAPISchemas:
    """Tests for Pydantic API schemas."""

    def test_text_input_validation(self):
        from api.schemas import TextInput

        # Valid input
        input_data = TextInput(text="Patient has diabetes mellitus type 2.")
        assert input_data.text == "Patient has diabetes mellitus type 2."
        assert input_data.generate_soap is False

    def test_text_input_too_short(self):
        from api.schemas import TextInput
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TextInput(text="Short")

    def test_health_response(self):
        from api.schemas import HealthResponse

        response = HealthResponse(
            status="healthy",
            models={"asr": True, "ner": True, "summarizer": False},
        )
        assert response.status == "healthy"
        assert response.models["asr"] is True


# =============================================================================
# FHIR Exporter Tests
# =============================================================================

class TestFHIRExporter:
    """Tests for HL7 FHIR R4 exporter."""

    def test_bundle_generation(self):
        from src.utils.fhir_exporter import FHIRR4Exporter

        exporter = FHIRR4Exporter()
        mock_output = {
            "transcript": "Patient with hypertension prescribed Lisinopril 10mg daily.",
            "entities": {
                "raw": [
                    {"text": "hypertension", "label": "Condition", "score": 0.95},
                    {"text": "lisinopril", "label": "Medication", "score": 0.98},
                    {"text": "10mg", "label": "Dosage", "score": 0.99},
                ]
            },
            "summary": {
                "soap": {
                    "S": "History of hypertension.",
                    "O": "BP 138/88.",
                    "A": "Hypertension under evaluation.",
                    "P": "Start Lisinopril 10mg."
                }
            }
        }

        bundle = exporter.to_bundle(mock_output, patient_id="PAT-999")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert bundle["total"] >= 3

        resource_types = [entry["resource"]["resourceType"] for entry in bundle["entry"]]
        assert "Encounter" in resource_types
        assert "DocumentReference" in resource_types
        assert "Condition" in resource_types
        assert "MedicationRequest" in resource_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
