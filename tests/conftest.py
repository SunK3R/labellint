# ==============================================================================
# tests.conftest: Centralized Test Fixtures for `labellint`
#
# This file defines shared Pytest fixtures used across the entire test suite.
# Centralizing test data and setup logic here provides several key benefits:
#
#   1. DRY Principle: Avoids duplicating complex data setup in multiple test
#      files. Tests become cleaner and more focused on their specific assertions.
#   2. Consistency: Ensures all tests run against the same, well-defined "golden"
#      data, preventing inconsistencies and hard-to-debug test failures.
#   3. Maintainability: If the core data model changes, we only need to update
#      the fixtures in this one file, rather than in dozens of individual tests.
#   4. Scalability: As the test suite grows, new tests can easily consume
#      these foundational fixtures without reinventing the wheel.
#
# Reference for fixture scopes: https://docs.pytest.org/en/latest/how-to/fixtures.html#scope-sharing-fixtures
# ==============================================================================

from pathlib import Path

import pytest

# Imports from the application itself.
from labellint.formats import (
    COCOAnnotation,
    COCOCategory,
    COCOData,
    COCOImage,
    COCOInfo,
    COCOLicense,
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """A session-scoped fixture that returns the project's root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def samples_dir(project_root: Path) -> Path:
    """Provides the path to the directory containing sample data files."""
    return project_root / "samples"


@pytest.fixture(scope="session")
def flawed_sample_path(samples_dir: Path) -> Path:
    """
    Provides the path to the canonical, intentionally flawed sample COCO file.
    This is the primary input for end-to-end integration tests.
    """
    path = samples_dir / "sample_coco.json"
    if not path.exists():
        pytest.fail(f"Critical test file not found: {path}")
    return path


@pytest.fixture(scope="function")
def clean_coco_data() -> COCOData:
    """
    Provides a pristine, in-memory `COCOData` object with no anomalies.

    This fixture has 'function' scope, meaning a fresh, mutable copy is
    generated for each test function. This isolation is crucial: it allows
    tests to modify the data (e.g., to introduce a specific flaw) without
    any risk of side effects impacting other tests.
    """
    return COCOData(
        info=COCOInfo(
            year=2024,
            version="1.0",
            description="Pristine COCO data for testing.",
            contributor="labellint.tests",
            url="http://labellint.dev",
            date_created="2024-01-01T00:00:00Z",
        ),
        licenses=[
            COCOLicense(id=1, name="Test License", url="http://labellint.dev/license")
        ],
        categories=[
            COCOCategory(id=1, name="car", supercategory="vehicle"),
            COCOCategory(id=2, name="person", supercategory="human"),
            COCOCategory(id=3, name="traffic light", supercategory="object"),
        ],
        images=[
            COCOImage(
                id=101,
                width=800,
                height=600,
                file_name="test_img_01.jpg",
                license=1,
            ),
            COCOImage(
                id=102,
                width=1920,
                height=1080,
                file_name="test_img_02.jpg",
                license=1,
            ),
        ],
        annotations=[
            # Annotations for image 101
            COCOAnnotation(
                id=1,
                image_id=101,
                category_id=1,
                bbox=[10, 10, 50, 50],
                area=2500.0,
                iscrowd=0,
            ),
            COCOAnnotation(
                id=2,
                image_id=101,
                category_id=2,
                bbox=[100, 100, 30, 80],
                area=2400.0,
                iscrowd=0,
            ),
            # Annotations for image 102
            COCOAnnotation(
                id=3,
                image_id=102,
                category_id=1,
                bbox=[200, 200, 150, 100],
                area=15000.0,
                iscrowd=0,
            ),
            COCOAnnotation(
                id=4,
                image_id=102,
                category_id=3,
                bbox=[500, 10, 20, 40],
                area=800.0,
                iscrowd=0,
            ),
        ],
    )


@pytest.fixture(scope="function")
def coco_data_factory(clean_coco_data: COCOData):
    """
    Provides a factory function to create modified `COCOData` objects.

    This is a more advanced pattern than directly modifying the `clean_coco_data`
    fixture. It allows tests to clearly state their intent by programmatically
    defining the data they need.

    Yields:
        A callable that can be used to generate `COCOData` objects.
    """

    def _factory(**kwargs) -> COCOData:
        """
        Creates a COCOData instance, using the clean data as a base.
        Any keyword arguments provided will overwrite the base data.
        """
        base_data = clean_coco_data.model_dump()
        base_data.update(kwargs)
        return COCOData.model_validate(base_data)

    return _factory


@pytest.fixture(scope="function")
def create_coco_file(tmp_path: Path):
    """
    A factory fixture to create a temporary COCO JSON file on disk.

    This is essential for testing file I/O operations in `formats.py` and
    for running end-to-end tests in `test_core.py` and `test_main.py` that
    require a file path as input.

    Args:
        tmp_path: The built-in pytest fixture for a temporary directory.

    Yields:
        A callable that takes a `COCOData` object and writes it to a file,
        returning the file's `Path` object.
    """

    def _creator(data: COCOData, filename: str = "test.json") -> Path:
        file_path = tmp_path / filename
        file_path.write_text(data.model_dump_json(indent=2))
        return file_path

    return _creator
