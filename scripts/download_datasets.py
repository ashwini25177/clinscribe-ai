"""
ClinScribe AI — Dataset Download Script
=========================================
Downloads and prepares publicly available datasets for all three modules.
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def download_asr_datasets(data_dir: str):
    """Download medical ASR datasets."""
    print("\n📥 Downloading ASR datasets...")
    asr_dir = ensure_dir(os.path.join(data_dir, "raw", "asr"))

    try:
        from datasets import load_dataset

        # Eka Medical ASR Evaluation Dataset
        print("   ⏳ Downloading Eka Medical ASR dataset...")
        dataset = load_dataset(
            "ekacare/eka-medical-asr-evaluation-dataset",
            cache_dir=os.path.join(asr_dir, "eka_medical_asr"),
        )
        print(f"   ✅ Eka Medical ASR: {dataset}")

    except Exception as e:
        print(f"   ⚠️  Error downloading ASR datasets: {e}")
        print("   💡 You can manually download from:")
        print("      - https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset")
        print("      - https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent")


def download_ner_datasets(data_dir: str):
    """Download clinical NER datasets."""
    print("\n📥 Downloading NER datasets...")
    ner_dir = ensure_dir(os.path.join(data_dir, "raw", "ner"))

    try:
        from datasets import load_dataset

        # NCBI Disease Corpus
        print("   ⏳ Downloading NCBI Disease Corpus...")
        ncbi_dataset = load_dataset(
            "ncbi_disease",
            cache_dir=os.path.join(ner_dir, "ncbi_disease"),
        )
        print(f"   ✅ NCBI Disease: {ncbi_dataset}")

        # BC5CDR (Chemical-Disease)
        print("   ⏳ Downloading BC5CDR dataset...")
        try:
            bc5cdr_dataset = load_dataset(
                "bigbio/bc5cdr",
                "bc5cdr_bigbio_kb",
                cache_dir=os.path.join(ner_dir, "bc5cdr"),
            )
            print(f"   ✅ BC5CDR: {bc5cdr_dataset}")
        except Exception as e:
            print(f"   ⚠️  BC5CDR download failed: {e}")
            print("   💡 Try: pip install bigbio")

        # JNLPBA
        print("   ⏳ Downloading JNLPBA dataset...")
        jnlpba_dataset = load_dataset(
            "jnlpba",
            cache_dir=os.path.join(ner_dir, "jnlpba"),
        )
        print(f"   ✅ JNLPBA: {jnlpba_dataset}")

    except Exception as e:
        print(f"   ⚠️  Error downloading NER datasets: {e}")
        print("   💡 Manually download from:")
        print("      - https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
        print("      - https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/")


def download_summarization_datasets(data_dir: str):
    """Download clinical summarization datasets."""
    print("\n📥 Downloading Summarization datasets...")
    summ_dir = ensure_dir(os.path.join(data_dir, "raw", "summarizer"))

    try:
        from datasets import load_dataset

        # PubMed Summarization (subset for development)
        print("   ⏳ Downloading PubMed Summarization (first 5000 samples)...")
        pubmed_dataset = load_dataset(
            "ccdv/pubmed-summarization",
            "document",
            split="train[:5000]",
            cache_dir=os.path.join(summ_dir, "pubmed_summarization"),
        )
        print(f"   ✅ PubMed Summarization: {len(pubmed_dataset)} samples")

    except Exception as e:
        print(f"   ⚠️  Error downloading summarization datasets: {e}")
        print("   💡 Manually download from:")
        print("      - https://huggingface.co/datasets/ccdv/pubmed-summarization")
        print("      - https://github.com/abachaa/MeQSum")


def create_sample_data(data_dir: str):
    """Create sample data files for development and testing."""
    import json

    print("\n📝 Creating sample data files...")

    # Sample NER data
    ner_sample_dir = ensure_dir(os.path.join(data_dir, "sample", "ner"))
    ner_samples = [
        {
            "tokens": ["Patient", "takes", "metformin", "1000mg", "twice", "daily", "for", "diabetes", "."],
            "ner_tags": ["O", "O", "B-DRUG", "B-DOSAGE", "B-FREQUENCY", "I-FREQUENCY", "O", "B-CONDITION", "O"],
        },
        {
            "tokens": ["Prescribed", "amoxicillin", "500mg", "oral", "three", "times", "daily", "for", "7", "days", "."],
            "ner_tags": ["O", "B-DRUG", "B-DOSAGE", "B-ROUTE", "B-FREQUENCY", "I-FREQUENCY", "I-FREQUENCY", "O", "B-DURATION", "I-DURATION", "O"],
        },
        {
            "tokens": ["The", "patient", "has", "hypertension", "and", "is", "on", "lisinopril", "10mg", "once", "daily", "."],
            "ner_tags": ["O", "O", "O", "B-CONDITION", "O", "O", "O", "B-DRUG", "B-DOSAGE", "B-FREQUENCY", "I-FREQUENCY", "O"],
        },
    ]
    with open(os.path.join(ner_sample_dir, "data.json"), "w") as f:
        json.dump(ner_samples, f, indent=2)
    print(f"   ✅ NER sample data: {len(ner_samples)} samples")

    # Sample summarization data
    summ_sample_dir = ensure_dir(os.path.join(data_dir, "sample", "summarizer"))
    from src.summarizer.dataset import create_clinical_summarization_samples

    summ_samples = create_clinical_summarization_samples()
    with open(os.path.join(summ_sample_dir, "data.json"), "w") as f:
        json.dump(summ_samples, f, indent=2)
    print(f"   ✅ Summarization sample data: {len(summ_samples)} samples")


def main():
    parser = argparse.ArgumentParser(description="Download ClinScribe AI datasets")
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Base data directory",
    )
    parser.add_argument(
        "--module", type=str, choices=["asr", "ner", "summarizer", "all"],
        default="all", help="Which module's data to download",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Only create sample data (no downloads)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  📦 ClinScribe AI — Dataset Downloader")
    print("=" * 60)

    if args.sample_only:
        create_sample_data(args.data_dir)
    else:
        if args.module in ("asr", "all"):
            download_asr_datasets(args.data_dir)
        if args.module in ("ner", "all"):
            download_ner_datasets(args.data_dir)
        if args.module in ("summarizer", "all"):
            download_summarization_datasets(args.data_dir)

        # Always create sample data
        create_sample_data(args.data_dir)

    print("\n✅ Dataset preparation complete!")
    print(f"   Data directory: {os.path.abspath(args.data_dir)}")


if __name__ == "__main__":
    main()
