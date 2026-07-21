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


class CandidateOut(BaseModel):
    class_id: int
    display_name: str
    similarity: float
    vlm_selected: Optional[bool] = False
    s_fused: Optional[float] = None
    exemplar_url: Optional[str] = None


class AnnotationOut(BaseModel):
    crop_id: str
    bbox: BBoxOut
    class_id: int
    confidence: float
    crop_data_url: Optional[str] = None
    parent_image_name: Optional[str] = None
    ocr_text: Optional[str] = None
    vlm_verified: Optional[bool] = None
    vlm_reason: Optional[str] = None
    commercial_sku: Optional[CommercialSKUOut] = None


class HITLRecordOut(BaseModel):
    hitl_id: str
    crop_id: str
    bbox: BBoxOut
    class_id: Optional[int] = None
    confidence: float
    reject_reason: str
    crop_data_url: Optional[str] = None
    parent_image_name: Optional[str] = None
    vlm_verified: Optional[bool] = None
    vlm_reason: Optional[str] = None
    commercial_sku: Optional[CommercialSKUOut] = None
    top5_candidates: Optional[List[CandidateOut]] = None


class AuditResponse(BaseModel):
    image_name: str
    parent_image_data_url: Optional[str] = None
    processing_time_ms: float = 0.0
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
    class_id: Optional[int] = None
    message: Optional[str] = None
