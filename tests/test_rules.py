# ==============================================================================
# tests.test_rules: Unit Tests for the Linter's Logic Core
#
# This module provides a comprehensive, high-coverage suite of unit tests for
# each function in `labellint.rules`.
#
# Test Design Philosophy:
#   - BDD-style Naming: Test classes are named `TestRule<Name>` and methods
#     use descriptive names like `test_on_clean_data` or `test_with_<flaw>`.
#   - Isolation: Each class tests exactly one rule.
#   - Positive & Negative Cases: Every rule is tested for both its "happy path"
#     (no findings on clean data) and its "sad path" (correctly identifying
#     specific, introduced flaws).
#   - Fixture-Driven: Tests consume fixtures from `conftest.py` to ensure they
#     operate on clean, isolated data, which is then modified as needed.
#     This prevents side effects between tests.
# ==============================================================================

import pytest
from labellint import rules
from labellint.formats import COCOAnnotation, COCOCategory, COCOData, COCOImage

# All tests in this file implicitly use the `clean_coco_data` fixture
# defined in `tests/conftest.py`.


class TestRuleCategoryCaseConsistency:
    """Tests for rules.check_category_case_consistency."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should find no issues in data with consistent capitalization."""
        findings = rules.check_category_case_consistency(clean_coco_data)
        assert not findings

    def test_with_inconsistent_case(self, clean_coco_data: COCOData) -> None:
        """Should detect categories that differ only by case."""
        clean_coco_data.categories.append(COCOCategory(id=4, name="Car"))
        clean_coco_data.categories.append(COCOCategory(id=5, name="PERSON"))
        findings = rules.check_category_case_consistency(clean_coco_data)
        assert len(findings) == 2
        assert "Inconsistent capitalization for 'car'. Found: Car, car" in findings
        assert (
            "Inconsistent capitalization for 'person'. Found: PERSON, person"
            in findings
        )


class TestRuleCategoryDuplicateNames:
    """Tests for rules.check_category_duplicate_names."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should find no issues when all category names are unique."""
        findings = rules.check_category_duplicate_names(clean_coco_data)
        assert not findings

    def test_with_duplicate_names(self, clean_coco_data: COCOData) -> None:
        """Should detect when the exact same name is used for multiple categories."""
        clean_coco_data.categories.append(COCOCategory(id=4, name="car"))
        findings = rules.check_category_duplicate_names(clean_coco_data)
        assert len(findings) == 1
        assert "Duplicate category name 'car' appears 2 times." in findings


class TestRuleCategoryDuplicateIds:
    """Tests for rules.check_category_duplicate_ids."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should find no issues when all category IDs are unique."""
        findings = rules.check_category_duplicate_ids(clean_coco_data)
        assert not findings

    def test_with_duplicate_ids(self, clean_coco_data: COCOData) -> None:
        """Should detect when the same ID is used for multiple categories."""
        clean_coco_data.categories.append(COCOCategory(id=1, name="truck"))
        findings = rules.check_category_duplicate_ids(clean_coco_data)
        assert len(findings) == 1
        assert "Duplicate category ID #1 appears 2 times." in findings


class TestRuleRelationUnmatchedAnnotations:
    """Tests for rules.check_relation_unmatched_annotations."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should find no issues when all annotations map to valid images."""
        findings = rules.check_relation_unmatched_annotations(clean_coco_data)
        assert not findings

    def test_with_orphaned_annotation(self, clean_coco_data: COCOData) -> None:
        """Should detect an annotation pointing to a non-existent image ID."""
        clean_coco_data.annotations.append(
            COCOAnnotation(
                id=99,
                image_id=9999,
                category_id=1,
                bbox=[0, 0, 1, 1],
                area=1,
                iscrowd=0,
            )
        )
        findings = rules.check_relation_unmatched_annotations(clean_coco_data)
        assert len(findings) == 1
        assert (
            "Orphaned annotation (ID 99) points to a missing image (ID 9999)."
            in findings[0]
        )


class TestRuleRelationUnmatchedCategory:
    """Tests for rules.check_relation_unmatched_category."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should find no issues when all annotations map to valid categories."""
        findings = rules.check_relation_unmatched_category(clean_coco_data)
        assert not findings

    def test_with_unmatched_category(self, clean_coco_data: COCOData) -> None:
        """Should detect an annotation pointing to a non-existent category ID."""
        clean_coco_data.annotations.append(
            COCOAnnotation(
                id=99,
                image_id=101,
                category_id=9999,
                bbox=[0, 0, 1, 1],
                area=1,
                iscrowd=0,
            )
        )
        findings = rules.check_relation_unmatched_category(clean_coco_data)
        assert len(findings) == 1
        assert (
            "Annotation (ID 99) points to a missing category (ID 9999)." in findings[0]
        )


class TestRuleRelationImagesWithoutAnnotations:
    """Tests for rules.check_relation_images_without_annotations."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should find no issues when all images have at least one annotation."""
        findings = rules.check_relation_images_without_annotations(clean_coco_data)
        assert not findings

    def test_with_unannotated_image(self, clean_coco_data: COCOData) -> None:
        """Should detect an image that has no corresponding annotations."""
        clean_coco_data.images.append(
            COCOImage(id=103, width=100, height=100, file_name="empty.jpg")
        )
        findings = rules.check_relation_images_without_annotations(clean_coco_data)
        assert len(findings) == 1
        assert "Image 'empty.jpg' (ID 103) has no annotations." in findings[0]

    def test_on_dataset_with_no_annotations(self, clean_coco_data: COCOData) -> None:
        """Should not flag anything if the entire dataset has zero annotations."""
        clean_coco_data.annotations = []
        findings = rules.check_relation_images_without_annotations(clean_coco_data)
        assert not findings


class TestRuleGeometryZeroAreaBboxes:
    """Tests for rules.check_geometry_zero_area_bboxes."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should not find any issues in bounding boxes with positive area."""
        findings = rules.check_geometry_zero_area_bboxes(clean_coco_data)
        assert not findings

    @pytest.mark.parametrize(
        "bad_bbox, expected_str",
        [
            ([10, 10, 0, 50], "[w=0.0, h=50.0]"),
            ([10, 10, 50, 0], "[w=50.0, h=0.0]"),
            ([10, 10, 0, 0], "[w=0.0, h=0.0]"),
        ],
    )
    def test_with_zero_area_box(
        self, clean_coco_data: COCOData, bad_bbox: list, expected_str: str
    ) -> None:
        """Should detect bounding boxes with zero width or height."""
        clean_coco_data.annotations.append(
            COCOAnnotation(
                id=99, image_id=101, category_id=1, bbox=bad_bbox, area=0, iscrowd=0
            )
        )
        findings = rules.check_geometry_zero_area_bboxes(clean_coco_data)
        assert len(findings) == 1
        assert "Annotation (ID 99)" in findings[0]
        assert expected_str in findings[0]


class TestRuleGeometryBboxOutOfBounds:
    """Tests for rules.check_geometry_bbox_out_of_bounds."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should not find issues when all bboxes are within image boundaries."""
        findings = rules.check_geometry_bbox_out_of_bounds(clean_coco_data)
        assert not findings

    @pytest.mark.parametrize(
        "bad_bbox, expected_str_part",
        [
            ([-10, 10, 50, 50], "out of bounds"),
            ([10, -10, 50, 50], "out of bounds"),
            ([780, 10, 50, 50], "x2=830.0"),
            ([10, 580, 50, 50], "y2=630.0"),
        ],
    )
    def test_with_out_of_bounds_box(
        self, clean_coco_data: COCOData, bad_bbox: list, expected_str_part: str
    ) -> None:
        """Should detect bounding boxes that extend beyond the image canvas."""
        clean_coco_data.annotations.append(
            COCOAnnotation(
                id=99, image_id=101, category_id=1, bbox=bad_bbox, area=2500, iscrowd=0
            )
        )
        findings = rules.check_geometry_bbox_out_of_bounds(clean_coco_data)
        assert len(findings) == 1
        assert "Annotation (ID 99)" in findings[0]
        assert expected_str_part in findings[0]

    def test_skips_annotations_with_unmatched_image_id(
        self, clean_coco_data: COCOData
    ) -> None:
        """
        Covers line 181 in rules.py.

        Ensures the rule does not crash and produces no findings for an
        out-of-bounds annotation that points to a non-existent image ID.
        """
        # This annotation has an out-of-bounds bbox AND an invalid image_id.
        clean_coco_data.annotations.append(
            COCOAnnotation(
                id=99,
                image_id=9999,
                category_id=1,
                bbox=[-10, -10, 50, 50],
                area=2500,
                iscrowd=0,
            )
        )
        # The rule should hit the `continue` on line 181 and produce no findings.
        findings = rules.check_geometry_bbox_out_of_bounds(clean_coco_data)
        assert not findings


class TestRuleAttributeAreaBboxMismatch:
    """Tests for rules.check_attribute_area_bbox_mismatch."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should not find issues when bbox area and 'area' attribute match."""
        findings = rules.check_attribute_area_bbox_mismatch(clean_coco_data)
        assert not findings

    def test_with_mismatched_area(self, clean_coco_data: COCOData) -> None:
        """Should detect significant discrepancies between w*h and the area field."""
        # Bbox area is 50*50=2500, but we set the area attribute to 9999.
        clean_coco_data.annotations.append(
            COCOAnnotation(
                id=99,
                image_id=101,
                category_id=1,
                bbox=[10, 10, 50, 50],
                area=9999,
                iscrowd=0,
            )
        )
        findings = rules.check_attribute_area_bbox_mismatch(clean_coco_data)
        assert len(findings) == 1
        assert "Annotation (ID 99) has a mismatched area." in findings[0]
        assert "Bbox area is 2500.00, but 'area' attribute is 9999.00." in findings[0]

    def test_skips_annotations_with_segmentation_data(
        self, clean_coco_data: COCOData
    ) -> None:
        """
        Covers the `continue` statement in check_attribute_area_bbox_mismatch.

        Ensures the rule correctly ignores annotations that have polygonal
        segmentation data, even if their bbox area and 'area' attribute mismatch,
        as this check is only intended for simple rectangular boxes.
        """
        # Arrange: Create an annotation with a non-empty segmentation list
        # AND a mismatched area. This combination is crucial.
        mismatched_polygonal_annotation = COCOAnnotation(
            id=999,
            image_id=101,
            category_id=1,
            bbox=[10, 10, 100, 100],  # Bbox area = 10000
            area=5000.0,  # Area is clearly mismatched
            iscrowd=0,
            segmentation=[
                [10, 10, 110, 10, 110, 110, 10, 110]
            ],  # This non-empty list triggers the 'continue'
        )
        clean_coco_data.annotations.append(mismatched_polygonal_annotation)

        # Act: Run the rule on the modified data
        findings = rules.check_attribute_area_bbox_mismatch(clean_coco_data)

        # Assert: The rule should have hit the `continue` and produced no findings,
        # correctly ignoring the polygonal annotation's mismatched area.
        assert not findings, "Rule should ignore annotations with segmentation data."


class TestRuleStatisticalBboxAspectRatioOutliers:
    """Tests for rules.check_statistical_bbox_aspect_ratio_outliers."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should not find outliers in a dataset with normal aspect ratios."""
        findings = rules.check_statistical_bbox_aspect_ratio_outliers(clean_coco_data)
        assert not findings

    def test_with_very_wide_box(self, clean_coco_data: COCOData) -> None:
        """Should identify an annotation with an extremely high aspect ratio."""
        clean_coco_data.annotations.append(
            COCOAnnotation(
                id=99,
                image_id=101,
                category_id=1,
                bbox=[10, 10, 500, 10],
                area=5000,
                iscrowd=0,
            )
        )  # AR = 50.0
        findings = rules.check_statistical_bbox_aspect_ratio_outliers(clean_coco_data)
        assert len(findings) == 1
        assert (
            "Annotation (ID 99)" in findings[0]
            and "outlier aspect ratio of 50.00" in findings[0]
        )

    def test_with_very_tall_box(self, clean_coco_data: COCOData) -> None:
        """Should identify an annotation with an extremely low aspect ratio."""
        # By removing the fixture's varied annotations and creating a tight cluster,
        # the outlier's status becomes unambiguous.
        clean_coco_data.annotations = []  # Start fresh for this specific test
        for i in range(20):
            # Create a tight cluster of boxes with aspect ratio ~1.0
            clean_coco_data.annotations.append(
                COCOAnnotation(
                    id=i,
                    image_id=101,
                    category_id=1,
                    bbox=[10, 10, 100, 100],
                    area=10000,
                    iscrowd=0,
                )
            )

        # Now add the extreme outlier.
        clean_coco_data.annotations.append(
            COCOAnnotation(
                id=98,
                image_id=101,
                category_id=1,
                bbox=[10, 10, 10, 500],
                area=5000,
                iscrowd=0,
            )
        )  # AR = 0.02

        findings = rules.check_statistical_bbox_aspect_ratio_outliers(clean_coco_data)

        assert len(findings) == 1
        assert (
            "Annotation (ID 98)" in findings[0]
            and "outlier aspect ratio of 0.02" in findings[0]
        )

    def test_on_data_with_no_valid_aspect_ratios(
        self, clean_coco_data: COCOData
    ) -> None:
        """Should handle data that has annotations but none with valid aspect ratios."""
        # Case 1: No annotations at all (covers line 197)
        clean_coco_data.annotations = []
        findings = rules.check_statistical_bbox_aspect_ratio_outliers(clean_coco_data)
        assert not findings, "Should not fail on dataset with no annotations"

        # Case 2: Only zero-area annotations (covers line 207)
        clean_coco_data.annotations = [
            COCOAnnotation(
                id=1,
                image_id=101,
                category_id=1,
                bbox=[10, 10, 0, 50],
                area=0,
                iscrowd=0,
            ),
            COCOAnnotation(
                id=2,
                image_id=101,
                category_id=1,
                bbox=[10, 10, 50, 0],
                area=0,
                iscrowd=0,
            ),
        ]
        findings = rules.check_statistical_bbox_aspect_ratio_outliers(clean_coco_data)
        assert not findings, "Should not fail on dataset with only zero-area boxes"


class TestRuleStatisticalClassDistributionImbalance:
    """Tests for rules.check_statistical_class_distribution_imbalance."""

    def test_on_clean_data(self, clean_coco_data: COCOData) -> None:
        """Should not flag imbalance in a reasonably balanced dataset."""
        # Add more annotations to make it balanced
        for i in range(10):
            clean_coco_data.annotations.append(
                COCOAnnotation(
                    id=10 + i,
                    image_id=101,
                    category_id=1,
                    bbox=[i, i, 1, 1],
                    area=1,
                    iscrowd=0,
                )
            )
        findings = rules.check_statistical_class_distribution_imbalance(clean_coco_data)
        assert not findings

    def test_with_rare_class(self, clean_coco_data: COCOData) -> None:
        """Should flag multiple rare classes in a large, imbalanced dataset."""
        # Add many annotations for one class to make others rare.
        for i in range(100):
            clean_coco_data.annotations.append(
                COCOAnnotation(
                    id=100 + i,
                    image_id=101,
                    category_id=1,
                    bbox=[i, i, 1, 1],
                    area=1,
                    iscrowd=0,
                )
            )

        findings = rules.check_statistical_class_distribution_imbalance(clean_coco_data)

        assert len(findings) == 2

        expected_findings = {
            "Severe class imbalance: Category 'person' has only 1 annotations.",
            "Severe class imbalance: Category 'traffic light' has only 1 annotations.",
        }
        assert set(findings) == expected_findings

    def test_robustness_against_unmatched_category_id(
        self, clean_coco_data: COCOData
    ) -> None:
        """Should not fail when a rare class ID does not exist in the categories list."""
        # Make category 1 overwhelmingly common
        for i in range(100):
            clean_coco_data.annotations.append(
                COCOAnnotation(
                    id=100 + i,
                    image_id=101,
                    category_id=1,
                    bbox=[i, i, 1, 1],
                    area=1,
                    iscrowd=0,
                )
            )

        # Add a few annotations for a non-existent category ID 999
        for i in range(3):
            clean_coco_data.annotations.append(
                COCOAnnotation(
                    id=200 + i,
                    image_id=101,
                    category_id=999,
                    bbox=[i, i, 1, 1],
                    area=1,
                    iscrowd=0,
                )
            )

        # This call would raise an AttributeError if the `else: continue` (line 248) did not exist.
        # The test passing proves the line is covered and the rule is robust.
        try:
            findings = rules.check_statistical_class_distribution_imbalance(
                clean_coco_data
            )
            # We only expect to see findings for the valid rare classes.
            # The non-existent category 999 should be silently ignored by this rule.
            assert len(findings) == 2  # for 'person' and 'traffic light'
        except AttributeError:
            pytest.fail(
                "Rule failed with AttributeError, indicating line 248 was not covered or is faulty."
            )
