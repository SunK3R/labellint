# ==============================================================================
# labellint.rules: The Linter's Logic Core
#
# This module contains the suite of validation functions ("rules") that form
# the heart of the `labellint` engine. Each rule is a pure, stateless function
# that takes a validated `COCOData` object and returns a list of string
# findings.
#
# Rule Design Principles:
#   1. Atomicity: Each rule checks for one specific type of anomaly.
#   2. Purity: Rules are deterministic and have no side effects. Their output
#      depends only on their input data.
#   3. Readability: Findings are formatted as clear, human-readable strings.
#   4. Performance: Rules should be efficient, using set operations and
#      indexed lookups where possible to handle large datasets.
#   5. Discoverability: Rules are prefixed with `check_` to be automatically
#      collected by the rule engine.
# ==============================================================================

import inspect
import sys
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Set

import numpy as np

from .formats import COCOData

# Define a type alias for a rule function for clarity.
Rule = Callable[[COCOData], List[str]]


# ==============================================================================
# Rule Discovery
# ==============================================================================


def get_all_rules() -> List[Rule]:
    """
    Discovers and returns all rule functions defined in this module.
    """
    current_module = sys.modules[__name__]
    rule_functions: List[Rule] = [
        obj
        for name, obj in inspect.getmembers(current_module)
        if inspect.isfunction(obj) and name.startswith("check_")
    ]
    rule_functions.sort(key=lambda f: f.__name__)
    return rule_functions


# ==============================================================================
# Category and Relational Rules
# ==============================================================================


def check_category_case_consistency(data: COCOData) -> List[str]:
    """Checks for inconsistent capitalization in category names."""
    findings: List[str] = []
    names_by_lower: Dict[str, Set[str]] = defaultdict(set)
    for category in data.categories:
        names_by_lower[category.name.lower()].add(category.name)

    for lower_name, original_names in names_by_lower.items():
        if len(original_names) > 1:
            sorted_names = sorted(list(original_names))
            finding = f"Inconsistent capitalization for '{lower_name}'. Found: {', '.join(sorted_names)}"
            findings.append(finding)
    return findings


def check_category_duplicate_names(data: COCOData) -> List[str]:
    """Finds multiple category definitions with the exact same name."""
    findings: List[str] = []
    name_counts = Counter(cat.name for cat in data.categories)
    for name, count in name_counts.items():
        if count > 1:
            finding = f"Duplicate category name '{name}' appears {count} times."
            findings.append(finding)
    return findings


def check_category_duplicate_ids(data: COCOData) -> List[str]:
    """Finds multiple category definitions with the same ID."""
    findings: List[str] = []
    id_counts = Counter(cat.id for cat in data.categories)
    for cat_id, count in id_counts.items():
        if count > 1:
            finding = f"Duplicate category ID #{cat_id} appears {count} times."
            findings.append(finding)
    return findings


def check_relation_unmatched_annotations(data: COCOData) -> List[str]:
    """Finds annotations that point to a non-existent image ID."""
    findings: List[str] = []
    valid_image_ids = {image.id for image in data.images}
    for ann in data.annotations:
        if ann.image_id not in valid_image_ids:
            finding = f"Orphaned annotation (ID {ann.id}) points to a missing image (ID {ann.image_id})."
            findings.append(finding)
    return findings


def check_relation_unmatched_category(data: COCOData) -> List[str]:
    """Finds annotations that point to a non-existent category ID."""
    findings: List[str] = []
    valid_category_ids = {cat.id for cat in data.categories}
    for ann in data.annotations:
        if ann.category_id not in valid_category_ids:
            finding = f"Annotation (ID {ann.id}) points to a missing category (ID {ann.category_id})."
            findings.append(finding)
    return findings


def check_relation_images_without_annotations(data: COCOData) -> List[str]:
    """Finds images that have no corresponding annotations."""
    if (
        not data.annotations
    ):  # If no annotations exist at all, this check is irrelevant.
        return []

    findings: List[str] = []
    annotated_image_ids = {ann.image_id for ann in data.annotations}

    for image in data.images:
        if image.id not in annotated_image_ids:
            finding = f"Image '{image.file_name}' (ID {image.id}) has no annotations."
            findings.append(finding)

    return findings


# ==============================================================================
# Annotation Geometry and Attribute Rules
# ==============================================================================


def check_geometry_zero_area_bboxes(data: COCOData) -> List[str]:
    """Identifies annotations with bounding boxes of zero width or height."""
    findings: List[str] = []
    for ann in data.annotations:
        _x, _y, w, h = ann.bbox
        if w == 0 or h == 0:
            finding = f"Annotation (ID {ann.id}) on image (ID {ann.image_id}) has a zero-area bounding box [w={w:.1f}, h={h:.1f}]."
            findings.append(finding)
    return findings


def check_geometry_bbox_out_of_bounds(data: COCOData) -> List[str]:
    """Identifies bounding boxes that extend beyond image dimensions."""
    findings: List[str] = []
    images_by_id = {image.id: image for image in data.images}
    for ann in data.annotations:
        if ann.image_id not in images_by_id:
            continue  # This is handled by the 'unmatched_annotations' rule.

        image = images_by_id[ann.image_id]
        x1, y1, w, h = ann.bbox
        x2, y2 = x1 + w, y1 + h

        if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
            finding = (
                f"Annotation (ID {ann.id}) on image '{image.file_name}' (ID {ann.image_id}) is out of bounds. "
                f"Bbox [x2={x2:.1f}, y2={y2:.1f}] vs. Image [w={image.width}, h={image.height}]."
            )
            findings.append(finding)
    return findings


def check_attribute_area_bbox_mismatch(data: COCOData) -> List[str]:
    """Finds significant mismatches between bbox area and the 'area' attribute."""
    findings: List[str] = []
    for ann in data.annotations:
        # not complex polygons. We check if segmentation is empty as a heuristic. ***
        if (
            ann.segmentation
            and isinstance(ann.segmentation, list)
            and len(ann.segmentation) > 0
        ):
            continue

        _x, _y, w, h = ann.bbox
        bbox_area = w * h
        # Use a relative tolerance to avoid flagging minor float precision issues.
        if not np.isclose(bbox_area, ann.area, rtol=1e-3):
            finding = (
                f"Annotation (ID {ann.id}) has a mismatched area. "
                f"Bbox area is {bbox_area:.2f}, but 'area' attribute is {ann.area:.2f}."
            )
            findings.append(finding)
    return findings


# ==============================================================================
# Statistical Anomaly Rules
# ==============================================================================


def check_statistical_bbox_aspect_ratio_outliers(data: COCOData) -> List[str]:
    """Identifies bounding boxes with extreme aspect ratios (outliers)."""
    if not data.annotations:
        return []

    findings: List[str] = []
    aspect_ratios = []
    for ann in data.annotations:
        _x, _y, w, h = ann.bbox
        if w > 0 and h > 0:
            aspect_ratios.append(w / h)

    if not aspect_ratios:
        return []

    # Use IQR method to find outliers, which is robust to non-normal distributions.
    q1 = np.percentile(aspect_ratios, 25)
    q3 = np.percentile(aspect_ratios, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    for ann in data.annotations:
        _x, _y, w, h = ann.bbox
        if w > 0 and h > 0:
            ar = w / h
            if not (lower_bound <= ar <= upper_bound):
                finding = (
                    f"Annotation (ID {ann.id}) on image (ID {ann.image_id}) has an outlier aspect ratio of {ar:.2f}. "
                    f"Typical range: [{lower_bound:.2f} - {upper_bound:.2f}]."
                )
                findings.append(finding)
    return findings


def check_statistical_class_distribution_imbalance(data: COCOData) -> List[str]:
    """Flags categories with very few annotations, indicating severe imbalance."""
    total_annotations = len(data.annotations)
    if not total_annotations or total_annotations < 50:
        return []

    findings: List[str] = []
    category_counts = Counter(ann.category_id for ann in data.annotations)
    categories_by_id = {cat.id: cat for cat in data.categories}

    threshold = max(10, total_annotations * 0.001)

    for cat_id, count in category_counts.items():
        if count < threshold:
            category = categories_by_id.get(cat_id)
            if category:
                category_name = category.name
                findings.append(
                    f"Severe class imbalance: Category '{category_name}' has only {count} annotations."
                )
            else:
                # This line is demonstrably covered by tests, but not visible
                # to the coverage tool on Python 3.9. We explicitly exclude it.
                continue  # pragma: no cover
    return findings
