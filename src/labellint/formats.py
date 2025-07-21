# ==============================================================================
# labellint.formats: Data Ingestion and Validation Layer
#
# This module is responsible for parsing and validating various annotation
# formats. It uses Pydantic to enforce a strict data schema, transforming
# raw file content into a reliable, standardized internal representation.
# This ensures the core engine operates on predictable and safe data structures.
# ==============================================================================

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

# --- Setup Internal Logger ---
# This logger is for internal debugging and tracing, not for user-facing output.
logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class ParseError(Exception):
    """Base exception for all parsing-related errors in this module."""

    def __init__(self, message: str, path: Optional[Path] = None):
        self.path = path
        super().__init__(message)


class FileAccessError(ParseError):
    """Raised when a file cannot be read due to permissions or existence issues."""


class InvalidFormatError(ParseError):
    """Raised when file content does not conform to the expected format schema."""


# ==============================================================================
# Pydantic Models for COCO Schema Validation (Object Detection)
#
# These models define the expected structure of a COCO annotations file for
# object detection and instance segmentation tasks. They serve as the single
# source of truth for what constitutes a valid dataset for the linter.
#
# Reference: https://cocodataset.org/#format-data
# ==============================================================================


class COCOInfo(BaseModel):
    """Pydantic model for the 'info' block in COCO."""

    model_config = ConfigDict(extra="allow")  # Be lenient with extra info fields

    year: Optional[int] = None
    version: Optional[str] = None
    description: Optional[str] = None
    contributor: Optional[str] = None
    url: Optional[str] = None
    date_created: Optional[str] = None


class COCOLicense(BaseModel):
    """Pydantic model for the 'licenses' block in COCO."""

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    url: Optional[str] = None


class COCOImage(BaseModel):
    """Pydantic model for an individual image record in COCO."""

    model_config = ConfigDict(extra="forbid")

    id: int
    width: int = Field(gt=0, description="Image width must be positive.")
    height: int = Field(gt=0, description="Image height must be positive.")
    file_name: str
    license: Optional[int] = None
    flickr_url: Optional[str] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None


class COCOAnnotation(BaseModel):
    """Pydantic model for a single annotation record in COCO."""

    model_config = ConfigDict(extra="forbid")

    id: int
    image_id: int
    category_id: int
    segmentation: Union[List[Any], Dict[str, Any]] = []
    area: float = Field(ge=0.0, description="Annotation area must be non-negative.")
    bbox: List[float] = Field(min_length=4, max_length=4)
    iscrowd: int = Field(ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def bbox_must_have_positive_dimensions(cls, v: List[float]) -> List[float]:
        """Validates that bounding box width and height are not negative."""
        _x, _y, w, h = v
        if w < 0 or h < 0:
            raise ValueError(
                f"Bounding box dimensions [width, height] must be non-negative. "
                f"Got [w={w}, h={h}]"
            )
        return v


class COCOCategory(BaseModel):
    """Pydantic model for a category record in COCO."""

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    supercategory: Optional[str] = None


class COCOData(BaseModel):
    """The root Pydantic model for a complete COCO object detection dataset."""

    model_config = ConfigDict(extra="forbid")  # Forbid any unexpected top-level keys

    info: COCOInfo
    licenses: List[COCOLicense]
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]


# ==============================================================================
# Public Parsing Function
# ==============================================================================


def parse_coco(filepath: str) -> COCOData:
    """
    Parses and validates a COCO JSON annotation file.

    This function performs a three-stage process:
    1.  Reads the file from disk, handling potential file system errors.
    2.  Parses the file content as JSON, handling JSON syntax errors.
    3.  Validates the parsed JSON against the strict `COCOData` Pydantic model.

    If all stages pass, it returns a validated `COCOData` object. Otherwise,
    it raises a specific `ParseError` subclass with a descriptive message.

    Args:
        filepath: The absolute path to the COCO JSON file.

    Returns:
        A validated `COCOData` object representing the annotation data.

    Raises:
        FileAccessError: If the file cannot be found or read.
        InvalidFormatError: If the file is not valid JSON or does not conform
                            to the COCO Pydantic schema.
    """
    path = Path(filepath)
    logger.info(f"Initiating COCO parse for: {path}")

    # --- Stage 1: File Reading ---
    try:
        logger.debug(f"Attempting to read file content from {path}.")
        raw_content = path.read_text(encoding="utf-8")
        logger.debug("File content read successfully.")
    except FileNotFoundError as e:
        msg = "The specified file does not exist."
        logger.error(f"{msg} Path: {path}")
        raise FileAccessError(msg, path=path) from e
    except OSError as e:
        msg = f"Could not read the file due to an OS-level error: {e}"
        logger.error(f"{msg} Path: {path}")
        raise FileAccessError(msg, path=path) from e
    except Exception as e:
        msg = f"An unexpected error occurred while reading the file: {e}"
        logger.error(f"{msg} Path: {path}")
        raise FileAccessError(msg, path=path) from e

    # --- Stage 2: JSON Parsing ---
    try:
        logger.debug("Attempting to parse file content as JSON.")
        data = json.loads(raw_content)
        logger.debug("JSON parsing successful.")
    except json.JSONDecodeError as e:
        msg = f"File is not a valid JSON document. Error at line {e.lineno}, column {e.colno}: {e.msg}"
        logger.error(f"JSON decoding failed for {path}: {msg}")
        raise InvalidFormatError(msg, path=path) from e

    # --- Stage 3: Pydantic Schema Validation ---
    try:
        logger.debug("Attempting to validate data against COCO Pydantic schema.")
        coco_data = COCOData.model_validate(data)
        logger.info(
            f"Validation successful. Found {len(coco_data.images)} images, "
            f"{len(coco_data.annotations)} annotations, "
            f"and {len(coco_data.categories)} categories."
        )
        return coco_data
    except ValidationError as e:
        # Pydantic's error messages are excellent but can be verbose.
        # We simplify the first error for a cleaner, more actionable user message.
        first_error = e.errors()[0]
        loc_str = " -> ".join(map(str, first_error["loc"]))
        msg = first_error["msg"]
        error_summary = f"Data validation failed at '{loc_str}': {msg}"

        logger.error(f"Pydantic validation failed for {path}: {error_summary}")
        logger.debug(f"Full Pydantic validation error:\n{e}")
        raise InvalidFormatError(error_summary, path=path) from e
