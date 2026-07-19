from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. Domain Data Transfer Objects (DTOs)
# ==========================================

class BBoxDTO(BaseModel):
    """Data transfer object representing object bounding box coordinates."""
    x1: float = Field(..., description="Top-left X coordinate.")
    y1: float = Field(..., description="Top-left Y coordinate.")
    x2: float = Field(..., description="Bottom-right X coordinate.")
    y2: float = Field(..., description="Bottom-right Y coordinate.")
    confidence: float = Field(..., description="Detector confidence score.")


class CropDTO(BaseModel):
    """Data transfer object representing an extracted product package crop."""
    crop_id: str = Field(..., description="Unique identifier for the crop.")
    image_bytes: bytes = Field(..., description="Raw crop image payload bytes (PNG/JPG).")
    bbox: BBoxDTO = Field(..., description="Bounding box location in the original shelf image.")
    blur_score: float = Field(..., description="Laplacian variance blur score.")
    aspect_ratio: float = Field(..., description="Width-to-height ratio.")


class EmbeddingDTO(BaseModel):
    """Data transfer object representing low-dimensional visual feature vectors."""
    vector: List[float] = Field(..., description="L2-normalized float vector.")
    dimension: int = Field(..., description="Vector dimensionality.")


class SearchResultDTO(BaseModel):
    """Data transfer object representing a database SKU match candidate."""
    remapped_class_id: int = Field(..., description="Remapped class target ID.")
    old_class_id: int = Field(..., description="Original class target ID.")
    similarity: float = Field(..., description="Cosine similarity score.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary.")


class OCRResultDTO(BaseModel):
    """Data transfer object representing text extracted from product packaging."""
    text: str = Field(..., description="Extracted raw text characters.")
    confidence: float = Field(..., description="OCR model confidence score.")


class CommercialSKUDTO(BaseModel):
    """Data transfer object representing commercial FMCG product catalog metadata."""
    project_sku_id: str = Field(..., description="Stable project SKU ID (e.g. TM_RAW_000).")
    display_name: str = Field(..., description="Full commercial product title.")
    brand: str = Field(..., description="Product brand name (e.g. Lipton).")
    product_name: str = Field(..., description="Product line name (e.g. Green Tea).")
    variant: str = Field(..., description="Product variant or flavor (e.g. Lemon).")
    pack_count: str = Field(..., description="Pack count or weight (e.g. 50 tea bags).")
    pack_type: str = Field(..., description="Packaging type (e.g. box, jar, bag).")


class PredictionDTO(BaseModel):
    """Data transfer object representing the final auto-annotation outcome."""
    bbox: BBoxDTO = Field(..., description="Bounding box coordinates.")
    predicted_class_id: int = Field(..., description="The classified SKU target ID.")
    confidence_probability: float = Field(..., description="Calibrated match probability.")
    automated: bool = Field(..., description="True if safe to auto-annotate without human verification.")
    reject_reason: Optional[str] = Field(None, description="Reason for HITL routing (e.g. LOW_CONFIDENCE).")
    ocr_text: Optional[str] = Field(None, description="Extracted OCR text if performed.")
    commercial_info: Optional[CommercialSKUDTO] = Field(None, description="Rich commercial SKU metadata.")


# ==========================================
# 2. Plugin Lifecycle Interface
# ==========================================

class IPlugin(ABC):
    """Interface enforcing production service lifecycle controls."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Loads weights, compiles graph engines, or initiates database connections."""
        pass

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """Runs diagnostics checks. Returns (status, error_message)."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Safely closes file locks, active requests, and DB pools."""
        pass


# ==========================================
# 3. Component Domain Interfaces
# ==========================================

class BaseDetector(IPlugin, ABC):
    """Abstract interface for shelf product localization engines."""

    @abstractmethod
    def detect(self, image_bytes: bytes) -> List[BBoxDTO]:
        """Detects packages in the image.

        Args:
            image_bytes: Raw shelf image bytes.

        Returns:
            List[BBoxDTO]: Detected bounding boxes.
        """
        pass


class BaseQualityGate(IPlugin, ABC):
    """Abstract interface for checking crop visual readability before matching."""

    @abstractmethod
    def is_valid(self, crop: CropDTO) -> Tuple[bool, str]:
        """Determines if crop quality permits retrieval search.

        Args:
            crop: Product crop object.

        Returns:
            Tuple[bool, str]: (is_valid, reject_reason).
        """
        pass


class BaseEmbedder(IPlugin, ABC):
    """Abstract interface for visual representation extractors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Expected feature vector length."""
        pass

    @abstractmethod
    def extract_dto(self, crop: CropDTO) -> EmbeddingDTO:
        """Extracts normalized visual feature vector for a single crop."""
        pass

    @abstractmethod
    def extract_batch_dto(self, crops: List[CropDTO]) -> List[EmbeddingDTO]:
        """Extracts features for a batch of crops safely."""
        pass


class BaseRetriever(IPlugin, ABC):
    """Abstract interface for nearest-neighbor similarity indices."""

    @abstractmethod
    def search_dto(
        self,
        embedding: EmbeddingDTO,
        top_k: int,
        family_id: Optional[str] = None
    ) -> List[SearchResultDTO]:
        """Finds closest candidates in the gallery registry."""
        pass


class BaseOCR(IPlugin, ABC):
    """Abstract interface for text extraction engines."""

    @abstractmethod
    def extract_text(self, crop: CropDTO, timeout_ms: int) -> OCRResultDTO:
        """Runs OCR on the crop within a timeout limit."""
        pass


class BaseCalibrator(IPlugin, ABC):
    """Abstract interface for similarity-to-probability mapping."""

    @abstractmethod
    def calibrate(self, similarity: float, class_id: int) -> float:
        """Converts similarity score to a true probability."""
        pass


class BaseFusionStrategy(IPlugin, ABC):
    """Abstract interface for combining visual similarities with text signals."""

    @abstractmethod
    def fuse(self, matches: List[SearchResultDTO], ocr: Optional[OCRResultDTO]) -> List[SearchResultDTO]:
        """Fuses visual similarity scores with OCR text matching scores."""
        pass


class BaseDecisionPolicy(IPlugin, ABC):
    """Abstract interface for routing predictions to HITL or Auto-Annotation."""

    @abstractmethod
    def decide(
        self,
        matches: List[SearchResultDTO],
        probability: float,
        class_id: int
    ) -> Tuple[bool, Optional[str]]:
        """Makes the automation decision.

        Returns:
            Tuple[bool, str]: (automated: bool, reject_reason: str).
        """
        pass


class BaseGalleryStore(IPlugin, ABC):
    """Abstract repository pattern interface for visual gallery reference storage."""

    @abstractmethod
    def save_reference(
        self,
        class_id: int,
        old_class_id: int,
        crop_path: str,
        family_id: str,
        source_image: str,
        bbox: BBoxDTO,
        embedding: EmbeddingDTO
    ) -> int:
        """Persists a reference crop embedding signature to the database.

        Returns:
            int: The incremented gallery database version.
        """
        pass

    @abstractmethod
    def fetch_all_references(self) -> Tuple[List[EmbeddingDTO], List[Dict[str, Any]]]:
        """Fetches all registered reference vectors and metadata."""
        pass

    @abstractmethod
    def delete_sku(self, class_id: int) -> int:
        """Deletes all references associated with a SKU class ID."""
        pass

    @abstractmethod
    def get_current_version(self) -> int:
        """Fetches current gallery database version integer."""
        pass

    @abstractmethod
    def rollback_version(self, version: int) -> None:
        """Rollbacks the database registry state to a previous version."""
        pass
