"""
FastAPI Request / Response Schemas
====================================
Pydantic models for type-safe API contracts.
"""

from typing import List, Dict
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left coordinate in pixels")
    y1: float = Field(..., description="Top coordinate in pixels")
    x2: float = Field(..., description="Right coordinate in pixels")
    y2: float = Field(..., description="Bottom coordinate in pixels")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detection(BaseModel):
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int = Field(..., ge=0)
    class_name: str


class DetectionResponse(BaseModel):
    """Response for single image /predict endpoint."""
    filename: str
    image_width: int
    image_height: int
    num_detections: int
    detections: List[Detection]
    inference_ms: float
    model_version: str
    use_sahi: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "satellite_001.jpg",
                "image_width": 4000,
                "image_height": 4000,
                "num_detections": 3,
                "detections": [
                    {
                        "bbox": {"x1": 1204.3, "y1": 892.1, "x2": 1256.8, "y2": 944.6},
                        "confidence": 0.87,
                        "class_id": 1,
                        "class_name": "ship"
                    }
                ],
                "inference_ms": 42.3,
                "model_version": "yolov8s-satdet-v1",
                "use_sahi": True
            }
        }


class BatchDetectionResponse(BaseModel):
    """Response for batch /predict_batch endpoint."""
    total_images: int
    total_detections: int
    results: List[DetectionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    num_classes: int
    class_names: Dict[int, str]
    input_size: int
    supports_sahi: bool
