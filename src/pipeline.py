"""
ClinScribe AI — End-to-End Pipeline
=====================================
Unified pipeline that chains:
  Audio → ASR (Whisper) → NER (BioBERT) → Summarization (Flan-T5) → JSON

This is the core orchestration module that connects all three AI components.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class ClinScribePipeline:
    """
    End-to-end clinical speech-to-insight pipeline.

    Pipeline stages:
    1. ASR: Audio → Text transcript
    2. NER: Text → Structured medical entities
    3. Summarization: Text → Clinical summary / SOAP note
    """

    def __init__(
        self,
        asr_model_path: str = "checkpoints/asr/final_model",
        ner_model_path: str = "checkpoints/ner/final_model",
        summarizer_model_path: str = "checkpoints/summarizer/final_model",
        device: Optional[str] = None,
        load_asr: bool = True,
        load_ner: bool = True,
        load_summarizer: bool = True,
    ):
        """
        Initialize the pipeline with all three models.

        Args:
            asr_model_path: Path to fine-tuned Whisper model.
            ner_model_path: Path to fine-tuned NER model.
            summarizer_model_path: Path to fine-tuned summarization model.
            device: Device for inference ('cuda', 'cpu', or None for auto).
            load_asr: Whether to load the ASR model.
            load_ner: Whether to load the NER model.
            load_summarizer: Whether to load the summarization model.
        """
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.asr_engine = None
        self.ner_engine = None
        self.summarizer_engine = None

        print("=" * 60)
        print("  🏥 ClinScribe AI — Initializing Pipeline")
        print("=" * 60)

        if load_asr:
            from src.asr.inference import MedicalASRInference

            self.asr_engine = MedicalASRInference(
                model_path=asr_model_path, device=self.device
            )

        if load_ner:
            from src.ner.inference import ClinicalNERInference

            self.ner_engine = ClinicalNERInference(
                model_path=ner_model_path, device=self.device
            )

        if load_summarizer:
            from src.summarizer.inference import ClinicalSummarizationInference

            self.summarizer_engine = ClinicalSummarizationInference(
                model_path=summarizer_model_path, device=self.device
            )

        print("\n✅ Pipeline initialized successfully!")
        print(f"   ASR: {'Loaded' if self.asr_engine else 'Skipped'}")
        print(f"   NER: {'Loaded' if self.ner_engine else 'Skipped'}")
        print(f"   Summarizer: {'Loaded' if self.summarizer_engine else 'Skipped'}")
        print("=" * 60)

    def process_audio(
        self,
        audio_path: str,
        language: str = "en",
        generate_soap: bool = False,
    ) -> Dict:
        """
        Process an audio file through the full pipeline.

        Audio → Transcript → Entities + Summary

        Args:
            audio_path: Path to the audio file.
            language: Language code for ASR.
            generate_soap: If True, generate SOAP-format summary.

        Returns:
            Complete analysis result dictionary.
        """
        start_time = time.time()
        result = {
            "input": {"audio_path": audio_path, "language": language},
            "transcript": None,
            "entities": None,
            "summary": None,
            "metadata": {},
        }

        # ── Stage 1: ASR ─────────────────────────────────────────────────────
        if self.asr_engine:
            print("\n🎤 Stage 1: Transcribing audio...")
            asr_result = self.asr_engine.transcribe(audio_path, language=language)
            result["transcript"] = asr_result["text"]
            result["metadata"]["asr"] = {
                "audio_duration": asr_result["audio_duration_seconds"],
                "processing_time": asr_result["processing_time_seconds"],
                "real_time_factor": asr_result["real_time_factor"],
            }
            print(f"   ✅ Transcription complete ({len(asr_result['text'].split())} words)")
        else:
            print("⚠️  ASR model not loaded. Provide text directly.")
            return result

        # Process the transcript through NER and summarization
        text_result = self.process_text(
            result["transcript"], generate_soap=generate_soap
        )

        result["entities"] = text_result["entities"]
        result["summary"] = text_result["summary"]
        result["metadata"].update(text_result.get("metadata", {}))

        # Total processing time
        total_time = time.time() - start_time
        result["metadata"]["total_processing_time"] = round(total_time, 3)

        return result

    def process_text(
        self,
        clinical_text: str,
        generate_soap: bool = False,
    ) -> Dict:
        """
        Process clinical text through NER and summarization.

        Useful when you already have a transcript and want to skip ASR.

        Args:
            clinical_text: Clinical text / transcript.
            generate_soap: If True, generate SOAP-format summary.

        Returns:
            Dictionary with entities and summary.
        """
        result = {
            "input_text": clinical_text,
            "entities": None,
            "summary": None,
            "metadata": {},
        }

        # ── Stage 2: NER ─────────────────────────────────────────────────────
        if self.ner_engine:
            print("\n🏷️  Stage 2: Extracting medical entities...")
            entities = self.ner_engine.extract_entities(clinical_text)
            structured = self.ner_engine.extract_structured(clinical_text)

            result["entities"] = {
                "raw": entities,
                "structured": structured,
                "total_found": len(entities),
            }
            result["metadata"]["ner"] = {"entities_found": len(entities)}
            print(f"   ✅ Found {len(entities)} entities")

        # ── Stage 3: Summarization ───────────────────────────────────────────
        if self.summarizer_engine:
            print("\n📝 Stage 3: Generating clinical summary...")
            if generate_soap:
                summary_result = self.summarizer_engine.generate_soap_note(
                    clinical_text
                )
            else:
                summary_result = self.summarizer_engine.summarize(clinical_text)

            result["summary"] = summary_result
            result["metadata"]["summarizer"] = {
                "compression_ratio": summary_result.get("compression_ratio"),
                "processing_time": summary_result.get("processing_time_seconds"),
            }
            print(f"   ✅ Summary generated ({summary_result.get('summary_length', 0)} words)")

        return result

    def process_batch(
        self,
        audio_paths: list = None,
        texts: list = None,
        **kwargs,
    ) -> list:
        """
        Process multiple audio files or texts through the pipeline.

        Args:
            audio_paths: List of audio file paths.
            texts: List of clinical text strings.
            **kwargs: Additional arguments.

        Returns:
            List of result dictionaries.
        """
        results = []

        if audio_paths:
            for i, path in enumerate(audio_paths):
                print(f"\n{'─' * 40}")
                print(f"  Processing [{i + 1}/{len(audio_paths)}]: {path}")
                result = self.process_audio(path, **kwargs)
                results.append(result)

        elif texts:
            for i, text in enumerate(texts):
                print(f"\n{'─' * 40}")
                print(f"  Processing [{i + 1}/{len(texts)}]")
                result = self.process_text(text, **kwargs)
                results.append(result)

        return results


def main():
    """CLI entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="ClinScribe AI — Clinical Speech-to-Insight Pipeline"
    )
    parser.add_argument(
        "--audio", type=str, help="Path to audio file"
    )
    parser.add_argument(
        "--text", type=str, help="Clinical text (alternative to audio)"
    )
    parser.add_argument(
        "--asr-model", type=str, default="checkpoints/asr/final_model",
        help="Path to ASR model",
    )
    parser.add_argument(
        "--ner-model", type=str, default="checkpoints/ner/final_model",
        help="Path to NER model",
    )
    parser.add_argument(
        "--summarizer-model", type=str, default="checkpoints/summarizer/final_model",
        help="Path to summarizer model",
    )
    parser.add_argument(
        "--soap", action="store_true", help="Generate SOAP note format"
    )
    parser.add_argument(
        "--output", type=str, help="Output JSON file path"
    )
    args = parser.parse_args()

    if not args.audio and not args.text:
        parser.error("Either --audio or --text must be provided")

    # Initialize pipeline
    pipeline = ClinScribePipeline(
        asr_model_path=args.asr_model,
        ner_model_path=args.ner_model,
        summarizer_model_path=args.summarizer_model,
        load_asr=args.audio is not None,
    )

    # Process
    if args.audio:
        result = pipeline.process_audio(args.audio, generate_soap=args.soap)
    else:
        result = pipeline.process_text(args.text, generate_soap=args.soap)

    # Output
    print("\n" + "=" * 60)
    print("  📊 ClinScribe AI — Results")
    print("=" * 60)

    if result.get("transcript"):
        print(f"\n📝 Transcript:\n{result['transcript']}")

    if result.get("entities"):
        entities = result["entities"]
        print(f"\n🏷️  Entities ({entities['total_found']}):")
        for entity in entities["raw"]:
            print(f"  [{entity['label']}] \"{entity['text']}\" (score: {entity['score']})")

    if result.get("summary"):
        summary = result["summary"]
        if "soap" in summary:
            print(f"\n📋 SOAP Note:")
            for section, content in summary["soap"].items():
                print(f"  {section}: {content}")
        else:
            print(f"\n📋 Summary:\n{summary.get('summary', '')}")

    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
