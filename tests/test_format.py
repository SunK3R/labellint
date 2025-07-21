# ==============================================================================
# tests.test_formats: Unit Tests for the Data Ingestion & Validation Layer
#
# This module provides a comprehensive suite of unit tests for the
# `labellint.formats` module. Its purpose is to validate the correctness of:
#   1. The Pydantic data models themselves (e.g., custom validators).
#   2. The `parse_coco` function's entire execution flow, especially its
#      three-stage error handling (File I/O -> JSON Parsing -> Schema Validation).
#
# Test Design Philosophy:
#   - White Box Testing: These tests are aware of the internal implementation
#     of `parse_coco` and are designed to exercise each specific error path.
#   - Exception Specificity: Tests assert that the *correct type* of custom
#     exception (`FileAccessError`, `InvalidFormatError`) is raised with the
#     expected error message.
#   - Fixture Usage: Leverages `tmp_path` and `create_coco_file` to test
#     interactions with the file system in a controlled, isolated environment.
# ==============================================================================

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from labellint.formats import (
    COCOAnnotation,
    COCOData,
    FileAccessError,
    InvalidFormatError,
    ParseError,
    parse_coco,
)
from pydantic import ValidationError

# All tests implicitly use fixtures from `tests/conftest.py`.


class TestPydanticModels:
    """
    Tests the validation logic built into the Pydantic models themselves.
    """

    def test_coco_data_forbids_extra_top_level_fields(self, clean_coco_data: COCOData):
        """
        GIVEN a valid COCOData object
        WHEN it is serialized and an extra top-level field is added
        THEN Pydantic validation should fail with a specific `extra_forbidden` error.
        """
        data_dict = clean_coco_data.model_dump()
        data_dict["unexpected_top_level_key"] = "some_value"

        with pytest.raises(ValidationError) as exc_info:
            COCOData.model_validate(data_dict)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_bbox",
        [
            ([10, 10, -50, 50]),  # Negative width
            ([10, 10, 50, -50]),  # Negative height
        ],
    )
    def test_coco_annotation_rejects_negative_bbox_dims(self, invalid_bbox: list):
        """
        GIVEN a bounding box with negative width or height
        WHEN an `COCOAnnotation` is created with it
        THEN Pydantic validation should raise a `ValidationError`.
        """
        with pytest.raises(ValidationError) as exc_info:
            COCOAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                bbox=invalid_bbox,
                area=0,
                iscrowd=0,
            )
        assert "must be non-negative" in str(exc_info.value)


class TestParseCocoHappyPath:
    """
    Tests the success path of the `parse_coco` function.
    """

    def test_parse_coco_on_valid_file(
        self, create_coco_file, clean_coco_data: COCOData
    ):
        """
        GIVEN a path to a well-formed and valid COCO JSON file
        WHEN `parse_coco` is called
        THEN it should return a valid `COCOData` Pydantic object.
        """
        # Arrange
        valid_file = create_coco_file(clean_coco_data)

        # Act
        result = parse_coco(str(valid_file))

        # Assert
        assert isinstance(result, COCOData)
        assert len(result.images) == len(clean_coco_data.images)
        assert len(result.annotations) == len(clean_coco_data.annotations)
        assert result.categories[0].name == "car"


class TestParseCocoErrorHandling:
    """
    Tests the specific error handling paths within the `parse_coco` function,
    covering all three stages of parsing failure.
    """

    # --- Stage 1: File I/O Errors ---
    def test_handles_file_not_found(self):
        """
        GIVEN a path that does not exist on the file system
        WHEN `parse_coco` is called
        THEN it should raise a `FileAccessError`.
        """
        with pytest.raises(FileAccessError, match="does not exist"):
            parse_coco("non/existent/path.json")

    @patch("pathlib.Path.read_text", side_effect=OSError("Permission denied."))
    def test_handles_io_error(self, mock_read_text):
        """
        GIVEN the OS raises an `IOError` during file read
        WHEN `parse_coco` is called
        THEN it should catch it and raise a `FileAccessError`.
        """
        with pytest.raises(FileAccessError, match="OS-level error"):
            parse_coco("dummy/path.json")
        mock_read_text.assert_called_once()

    @patch(
        "pathlib.Path.read_text", side_effect=Exception("A generic, unexpected error.")
    )
    def test_handles_unexpected_read_error(self, mock_read_text):
        """
        GIVEN an unexpected exception occurs during file read
        WHEN `parse_coco` is called
        THEN it should catch it and raise a `FileAccessError`.
        """
        with pytest.raises(FileAccessError, match="unexpected error occurred"):
            parse_coco("dummy/path.json")
        mock_read_text.assert_called_once()

    # --- Stage 2: JSON Parsing Errors ---
    def test_handles_malformed_json(self, tmp_path: Path):
        """
        GIVEN a file that is not valid JSON
        WHEN `parse_coco` is called
        THEN it should raise an `InvalidFormatError`.
        """
        malformed_file = tmp_path / "bad.json"
        malformed_file.write_text(
            "{'key': 'value',}"
        )  # Contains trailing comma and single quotes

        with pytest.raises(InvalidFormatError, match="not a valid JSON document"):
            parse_coco(str(malformed_file))

    # --- Stage 3: Pydantic Schema Validation Errors ---
    @pytest.mark.parametrize(
        "field_to_break, expected_error_msg",
        [
            ("images", "Input should be a valid list"),  # Break list type
            ("annotations", "Input should be a valid list"),
            ("categories", "Input should be a valid list"),
            ("licenses", "Input should be a valid list"),
            ("info", "Input should be a valid dictionary"),  # Break dict type
        ],
    )
    def test_handles_schema_type_mismatch(
        self,
        clean_coco_data: COCOData,
        create_coco_file,
        field_to_break: str,
        expected_error_msg: str,
    ):
        """
        GIVEN a file with a structurally incorrect COCO schema (e.g., a top-level key is the wrong type)
        WHEN `parse_coco` is called
        THEN it should raise an `InvalidFormatError` with a specific message.
        """
        # Arrange
        data_dict = clean_coco_data.model_dump()
        data_dict[field_to_break] = (
            "this is the wrong type"  # Intentionally break the schema
        )
        invalid_file = create_coco_file(
            COCOData.model_validate(clean_coco_data), "invalid.json"
        )
        invalid_file.write_text(json.dumps(data_dict))  # Write the broken version

        # Act & Assert
        with pytest.raises(InvalidFormatError) as exc_info:
            parse_coco(str(invalid_file))

        assert f"Data validation failed at '{field_to_break}'" in str(exc_info.value)
        assert expected_error_msg in str(exc_info.value)

    def test_handles_missing_required_field(
        self, clean_coco_data: COCOData, create_coco_file
    ):
        """
        GIVEN a file with a missing required field (e.g., `file_name` in an image)
        WHEN `parse_coco` is called
        THEN it should raise an `InvalidFormatError`.
        """
        # Arrange
        data_dict = clean_coco_data.model_dump()
        del data_dict["images"][0]["file_name"]  # Remove a required field
        invalid_file = create_coco_file(
            COCOData.model_validate(clean_coco_data), "invalid.json"
        )
        invalid_file.write_text(json.dumps(data_dict))

        # Act & Assert
        with pytest.raises(InvalidFormatError) as exc_info:
            parse_coco(str(invalid_file))

        assert "Data validation failed at 'images -> 0 -> file_name'" in str(
            exc_info.value
        )
        assert "Field required" in str(exc_info.value)

    def test_exception_inheritance(self):
        """
        GIVEN the custom exception types
        WHEN they are instantiated
        THEN they should inherit from the base `ParseError`.
        """
        assert issubclass(FileAccessError, ParseError)
        assert issubclass(InvalidFormatError, ParseError)
