from pydantic import BaseModel, Field
from typing import List, Optional


class BBoxOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class CommercialSKUOut(BaseModel):
    project_sku_id: str
    display_name: str
    brand: str
    product_name: str
    variant: str
    pack_count: str
    pack_type: str


class AnnotationOut(BaseModel):
    bbox: BBoxOut
    class_id: int
    confidence: float
    ocr_text: Optional[str] = None
    commercial_sku: Optional[CommercialSKUOut] = None


class HITLRecordOut(BaseModel):
    bbox: BBoxOut
    class_id: Optional[int] = None
    confidence: float
    reject_reason: str
    commercial_sku: Optional[CommercialSKUOut] = None


class AuditResponse(BaseModel):
    image_name: str
    annotations: List[AnnotationOut]
    hitl_queue: List[HITLRecordOut]


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    db_version: int


class OnboardResponse(BaseModel):
    status: str
    version: int
    crops_added: int
