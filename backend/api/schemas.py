from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class PronunciationRequest(BaseModel):
    target_word: Optional[str] = None
    target_phonemes: Optional[str] = None

class ErrorStats(BaseModel):
    sub: int
    ins: int
    deletions: int = Field(alias="del")

class PronunciationResponse(BaseModel):
    scores: Dict[str, Any]
    analysis: Dict[str, Any]
    feedback: Optional[str] = None
