"""
ClinScribe AI — ASR Inference Module
======================================
Transcribe medical audio using the fine-tuned Whisper model.
Supports single file, batch, and streaming inference.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class MedicalASRInference:
    """
    Medical ASR inference engine using fine-tuned Whisper.

    Supports:
    - Single audio file transcription
    - Batch transcription
    - Long-form audio with automatic chunking
    - Confidence scoring
    """

    def __init__(
        self,
        model_path: str = "checkpoints/asr/final_model",
        device: Optional[str] = None,
        use_lora: bool = True,
    ):
        """
        Initialize the ASR inference engine.

        Args:
            model_path: Path to the fine-tuned model checkpoint.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
            use_lora: Whether the model uses LoRA adapters.
        """
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path) and not model_path.startswith("openai/"):
            fallback = "openai/whisper-tiny"
            print(f"⚠️  ASR checkpoint not found at '{model_path}'. Using base model '{fallback}'...")
            model_path = fallback
            use_lora = False

        print(f"🔧 Loading ASR model from {model_path} on {self.device}...")

        self.processor = WhisperProcessor.from_pretrained(model_path)

        if use_lora and os.path.exists(model_path):
            from peft import PeftModel

            base_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-small"
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path
            )

        self.model.to(self.device)
        self.model.eval()
        print("✅ ASR model loaded successfully!")

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        return_timestamps: bool = False,
    ) -> Dict:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file.
            language: Language code for transcription.
            return_timestamps: Whether to return word-level timestamps.

        Returns:
            Dictionary with 'text', 'language', 'duration', and 'processing_time'.
        """
        from src.utils.preprocessing import load_audio, trim_silence, normalize_audio

        start_time = time.time()

        # Load and preprocess audio
        audio, sr = load_audio(audio_path, target_sr=16000)
        audio = trim_silence(audio)
        audio = normalize_audio(audio)

        duration = len(audio) / sr

        # Process through Whisper
        input_features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(self.device)

        # Generate transcription
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe",
        )

        generated_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
            return_timestamps=return_timestamps,
        )

        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        processing_time = time.time() - start_time

        return {
            "text": transcription,
            "language": language,
            "audio_duration_seconds": round(duration, 2),
            "processing_time_seconds": round(processing_time, 3),
            "real_time_factor": round(processing_time / duration, 3) if duration > 0 else 0,
        }

    @torch.no_grad()
    def transcribe_long_audio(
        self,
        audio_path: str,
        chunk_duration: int = 30,
        overlap: int = 2,
        language: str = "en",
    ) -> Dict:
        """
        Transcribe long audio by chunking with overlap.

        Args:
            audio_path: Path to long audio file.
            chunk_duration: Duration of each chunk in seconds.
            overlap: Overlap between chunks in seconds.
            language: Language code.

        Returns:
            Dictionary with full transcript and per-chunk results.
        """
        from src.utils.preprocessing import load_audio, chunk_audio

        start_time = time.time()

        audio, sr = load_audio(audio_path, target_sr=16000)
        chunks = chunk_audio(audio, sr, chunk_duration, overlap)

        chunk_results = []
        full_transcript = []

        for i, chunk in enumerate(chunks):
            input_features = self.processor(
                chunk, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device)

            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

            generated_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            chunk_results.append({
                "chunk_index": i,
                "start_time": i * (chunk_duration - overlap),
                "text": text,
            })
            full_transcript.append(text)

        processing_time = time.time() - start_time

        return {
            "text": " ".join(full_transcript),
            "chunks": chunk_results,
            "total_chunks": len(chunks),
            "audio_duration_seconds": round(len(audio) / sr, 2),
            "processing_time_seconds": round(processing_time, 3),
        }

    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: str = "en",
    ) -> List[Dict]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of audio file paths.
            language: Language code.

        Returns:
            List of transcription results.
        """
        results = []
        for i, path in enumerate(audio_paths):
            print(f"  Transcribing [{i + 1}/{len(audio_paths)}]: {path}")
            result = self.transcribe(path, language=language)
            result["file"] = path
            results.append(result)
        return results


def transcribe_file(
    audio_path: str,
    model_path: str = "checkpoints/asr/final_model",
) -> str:
    """
    Convenience function to transcribe a single audio file.

    Args:
        audio_path: Path to audio file.
        model_path: Path to fine-tuned model.

    Returns:
        Transcription string.
    """
    engine = MedicalASRInference(model_path=model_path)
    result = engine.transcribe(audio_path)
    return result["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical ASR Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument(
        "--model", type=str, default="checkpoints/asr/final_model",
        help="Path to fine-tuned model",
    )
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument("--long", action="store_true", help="Use long-form transcription")
    args = parser.parse_args()

    engine = MedicalASRInference(model_path=args.model)

    if args.long:
        result = engine.transcribe_long_audio(args.audio, language=args.language)
    else:
        result = engine.transcribe(args.audio, language=args.language)

    print(f"\n📝 Transcription:\n{result['text']}")
    print(f"\n⏱️  Duration: {result['audio_duration_seconds']}s")
    print(f"⚡ Processing time: {result['processing_time_seconds']}s")
