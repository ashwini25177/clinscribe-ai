"""
ClinScribe AI — FastAPI Inference Server
==========================================
REST API for the clinical speech-to-insight pipeline.

Endpoints:
  POST /analyze          - Full pipeline (audio → entities + summary)
  POST /transcribe       - ASR only (audio → text)
  POST /extract-entities - NER only (text → entities)
  POST /summarize        - Summarization only (text → summary)
  GET  /health           - Health check
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.schemas import (
    TextInput,
    TranscriptionResponse,
    EntityResponse,
    SummaryResponse,
    FullAnalysisResponse,
    HealthResponse,
)
from src.utils.fhir_exporter import FHIRR4Exporter

# ==============================================================================
# App Configuration
# ==============================================================================

app = FastAPI(
    title="ClinScribe AI",
    description=(
        "Clinical Speech-to-Insight Pipeline API. "
        "Transcribes medical audio, extracts clinical entities, "
        "and generates structured summaries."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded)
_pipeline = None


def get_pipeline():
    """Lazy-load the pipeline to avoid slow startup."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import ClinScribePipeline

        _pipeline = ClinScribePipeline(
            asr_model_path=os.environ.get(
                "ASR_MODEL_PATH", "checkpoints/asr/final_model"
            ),
            ner_model_path=os.environ.get(
                "NER_MODEL_PATH", "checkpoints/ner/final_model"
            ),
            summarizer_model_path=os.environ.get(
                "SUMMARIZER_MODEL_PATH", "checkpoints/summarizer/final_model"
            ),
        )
    return _pipeline


# ==============================================================================
# Endpoints
# ==============================================================================


@app.get("/", response_class=FileResponse, tags=["Dashboard"])
async def serve_dashboard():
    """Serve the ClinScribe AI interactive web dashboard."""
    index_file = Path(PROJECT_ROOT) / "web" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI file not found")
    return FileResponse(index_file)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and model health status."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        models={
            "asr": pipeline.asr_engine is not None,
            "ner": pipeline.ner_engine is not None,
            "summarizer": pipeline.summarizer_engine is not None,
        },
    )


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Pipeline"])
async def analyze_audio(
    audio: UploadFile = File(...),
    language: str = "en",
    generate_soap: bool = False,
):
    """
    Full pipeline analysis: Audio → Transcript + Entities + Summary.

    Upload a clinical audio file (.wav, .mp3, .flac) and receive:
    - Transcription
    - Extracted medical entities
    - Clinical summary or SOAP note
    """
    # Validate file type
    allowed_types = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    file_ext = Path(audio.filename).suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_types}",
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        suffix=file_ext, delete=False
    ) as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        pipeline = get_pipeline()
        result = pipeline.process_audio(
            tmp_path,
            language=language,
            generate_soap=generate_soap,
        )

        return FullAnalysisResponse(
            transcript=result.get("transcript", ""),
            entities=result.get("entities", {}).get("raw", []),
            structured_entities=result.get("entities", {}).get("structured", {}),
            summary=result.get("summary", {}).get("summary", ""),
            soap=result.get("summary", {}).get("soap"),
            metadata=result.get("metadata", {}),
        )
    finally:
        os.unlink(tmp_path)


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["ASR"])
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = "en",
):
    """Transcribe clinical audio to text using fine-tuned Whisper."""
    file_ext = Path(audio.filename).suffix.lower()

    with tempfile.NamedTemporaryFile(
        suffix=file_ext, delete=False
    ) as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        pipeline = get_pipeline()
        if not pipeline.asr_engine:
            raise HTTPException(status_code=503, detail="ASR model not loaded")

        result = pipeline.asr_engine.transcribe(tmp_path, language=language)
        return TranscriptionResponse(**result)
    finally:
        os.unlink(tmp_path)


@app.post("/extract-entities", response_model=EntityResponse, tags=["NER"])
async def extract_entities(input_data: TextInput):
    """Extract medical entities from clinical text."""
    pipeline = get_pipeline()
    if not pipeline.ner_engine:
        raise HTTPException(status_code=503, detail="NER model not loaded")

    entities = pipeline.ner_engine.extract_entities(input_data.text)
    structured = pipeline.ner_engine.extract_structured(input_data.text)

    return EntityResponse(
        entities=entities,
        structured=structured,
        total_found=len(entities),
    )


@app.post("/summarize", response_model=SummaryResponse, tags=["Summarization"])
async def summarize_text(input_data: TextInput):
    """Generate a clinical summary from text."""
    pipeline = get_pipeline()
    if not pipeline.summarizer_engine:
        raise HTTPException(status_code=503, detail="Summarizer model not loaded")

    if input_data.generate_soap:
        result = pipeline.summarizer_engine.generate_soap_note(input_data.text)
    else:
        result = pipeline.summarizer_engine.summarize(input_data.text)

    return SummaryResponse(**result)


@app.post("/export-fhir", tags=["Interoperability"])
async def export_fhir_bundle(input_data: TextInput):
    """
    Process clinical text and export as an HL7 FHIR Release 4 Bundle.
    Includes Encounter, DocumentReference (SOAP note), Condition, and MedicationRequest resources.
    """
    pipeline = get_pipeline()
    result = pipeline.process_text(input_data.text, generate_soap=True)
    exporter = FHIRR4Exporter()
    fhir_bundle = exporter.to_bundle(result)
    return fhir_bundle


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
