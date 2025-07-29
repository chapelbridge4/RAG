from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    use_hyde: bool = Field(default=True, description="Enable HyDE query expansion")
    max_corrections: Optional[int] = Field(default=2, description="Maximum self-corrections")

class ValidationScores(BaseModel):
    groundedness_score: float
    relevance_score: float
    accuracy_score: float
    overall_quality: float
    needs_correction: bool
    confidence_level: str

class QueryResponse(BaseModel):
    query: str
    response: str
    processing_time: float
    validation_scores: ValidationScores
    corrections_made: int
    metadata: Dict[str, Any]

class DocumentUpload(BaseModel):
    filename: str
    content: str
    metadata: Optional[Dict[str, Any]] = None