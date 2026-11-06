"""
ClinScribe AI — API Schemas (Pydantic Models)
===============================================
Request and response models for the FastAPI inference server.
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# ==============================================================================
# Request Models
# ==============================================================================

class TextInput(BaseModel):
    """Request body for text-based endpoints."""

    text: str = Field(
        ...,
        description="Clinical text to analyze",
        min_length=10,
        examples=[
            "Patient presents with persistent cough for two weeks. "
            "Started on amoxicillin 500mg three times daily."
        ],
    )
    generate_soap: bool = Field(
        default=False,
        description="If True, generate a SOAP-format clinical note",
    )


# ==============================================================================
# Response Models
# ==============================================================================

class TranscriptionResponse(BaseModel):
    """Response for ASR transcription endpoint."""

    text: str = Field(..., description="Transcribed text")
    language: str = Field(default="en", description="Detected/specified language")
    audio_duration_seconds: float = Field(
        ..., description="Duration of the audio file in seconds"
    )
    processing_time_seconds: float = Field(
        ..., description="Time taken for transcription"
    )
    real_time_factor: float = Field(
        ..., description="Processing time / audio duration"
    )


class EntityItem(BaseModel):
    """A single extracted entity."""

    text: str = Field(..., description="Entity text span")
    label: str = Field(..., description="Entity type (DRUG, CONDITION, etc.)")
    start: int = Field(..., description="Start character offset")
    end: int = Field(..., description="End character offset")
    score: float = Field(..., description="Confidence score")


class EntityResponse(BaseModel):
    """Response for NER entity extraction endpoint."""

    entities: List[Dict[str, Any]] = Field(
        ..., description="List of extracted entities"
    )
    structured: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Entities organized by category",
    )
    total_found: int = Field(..., description="Total number of entities found")


class SummaryResponse(BaseModel):
    """Response for summarization endpoint."""

    summary: str = Field(..., description="Generated clinical summary")
    input_length: int = Field(..., description="Input text word count")
    summary_length: int = Field(..., description="Summary word count")
    compression_ratio: float = Field(
        ..., description="Summary length / input length"
    )
    processing_time_seconds: float = Field(
        ..., description="Time taken for summarization"
    )
    soap: Optional[Dict[str, str]] = Field(
        default=None,
        description="SOAP note sections (S, O, A, P) if requested",
    )


class FullAnalysisResponse(BaseModel):
    """Response for the full pipeline analysis endpoint."""

    transcript: str = Field(..., description="Transcribed text from audio")
    entities: List[Dict[str, Any]] = Field(
        ..., description="Extracted medical entities"
    )
    structured_entities: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Entities organized by category",
    )
    summary: str = Field(..., description="Generated clinical summary")
    soap: Optional[Dict[str, str]] = Field(
        default=None,
        description="SOAP note sections if requested",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing metadata (timings, counts, etc.)",
    )


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = Field(..., description="System status")
    models: Dict[str, bool] = Field(
        ..., description="Model loading status"
    )
