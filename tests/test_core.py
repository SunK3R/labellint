# ==============================================================================
# tests.test_core: Unit & Integration Tests for the Linter Engine
#
# This module provides a comprehensive test suite for `labellint.core.run_scan`.
# It verifies that the engine correctly orchestrates the parsing and rule
# execution pipeline under various conditions, including success, failure,
# and edge cases.
#
# Test Design Philosophy:
#   - Layered Testing: We use a mix of end-to-end tests (on real files) to
#     validate the entire pipeline and mocked unit tests to isolate specific
#     behaviors like error handling and result aggregation.
#   - Log Validation: The `caplog` fixture is used extensively to assert that
#     the engine's internal logging provides a correct and transparent record
#     of its execution path.
#   - Exception Correctness: Tests assert not only that exceptions are raised,
#     but that the *correct type* of exception is raised, ensuring a stable API.
# ==============================================================================

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from labellint import core
from labellint.formats import COCOData, InvalidFormatError, ParseError

# All tests implicitly use fixtures defined in `tests/conftest.py`
# such as `flawed_sample_path`, `clean_coco_data`, and `create_coco_file`.


class TestCoreRunScanEndToEnd:
    """
    Tests the `run_scan` function from end-to-end, using real files and rules.
    This class validates the primary success paths of the application.
    """

    def test_on_flawed_sample(self, flawed_sample_path: Path, caplog):
        """
        GIVEN a path to the canonical flawed COCO sample file
        WHEN `run_scan` is called
        THEN it should return a ScanResult with all expected findings.
        """
        caplog.set_level(logging.INFO)
        result = core.run_scan(str(flawed_sample_path))

        assert isinstance(result, dict)
        assert isinstance(result["data"], COCOData)

        # Flaws: case(1), dupe_id(1), unmatched_ann(1), unmatched_cat(1),
        # no_ann_img(1), zero_area(2), out_of_bounds(2), area_mismatch(1),
        # aspect_ratio(1), class_imbalance(not run due to guard) = 11 total.
        expected_total_findings = 11
        expected_rules_with_findings = 9  # Number of keys in the findings dict
        assert result["total_findings"] == expected_total_findings
        assert len(result["findings"]) == expected_rules_with_findings

        assert "check_category_case_consistency" in result["findings"]
        assert "check_geometry_bbox_out_of_bounds" in result["findings"]
        # The imbalance rule should NOT be in the findings for our small sample
        assert (
            "check_statistical_class_distribution_imbalance" not in result["findings"]
        )
        assert (
            f"Scan complete. Total findings: {expected_total_findings}" in caplog.text
        )

    def test_on_clean_data(self, create_coco_file, clean_coco_data: COCOData, caplog):
        """
        GIVEN a path to a perfectly valid COCO file with no flaws
        WHEN `run_scan` is called
        THEN it should return a ScanResult with zero findings.
        """
        caplog.set_level(logging.INFO)
        clean_file_path = create_coco_file(clean_coco_data)
        result = core.run_scan(str(clean_file_path))

        assert result["total_findings"] == 0
        assert not result["findings"]
        assert "Scan complete. Total findings: 0" in caplog.text


class TestCoreRunScanErrorHandling:
    """
    Tests the `run_scan` function's error handling and fault tolerance.
    """

    def test_bubbles_up_parse_error_on_file_not_found(self, caplog):
        """
        GIVEN a path to a non-existent file
        WHEN `run_scan` is called
        THEN it should propagate a `ParseError` and log the failure.
        """
        caplog.set_level(logging.ERROR)
        non_existent_path = "path/to/a/file/that/does/not/exist.json"

        # Act & Assert
        with pytest.raises(ParseError):
            core.run_scan(non_existent_path)

        assert "Core engine caught a parsing error. Aborting scan." in caplog.text

    def test_bubbles_up_invalid_format_error(self, tmp_path: Path, caplog):
        """
        GIVEN a path to a file with malformed JSON
        WHEN `run_scan` is called
        THEN it should propagate an `InvalidFormatError`.
        """
        caplog.set_level(logging.ERROR)
        malformed_file = tmp_path / "bad.json"
        malformed_file.write_text("{'this is not valid json',}")

        with pytest.raises(InvalidFormatError):
            core.run_scan(str(malformed_file))

        assert "Core engine caught a parsing error. Aborting scan." in caplog.text

    @patch("labellint.core.rules.get_all_rules")
    @patch("labellint.core.formats.parse_coco")
    def test_handles_individual_rule_failure_gracefully(
        self, mock_parse_coco, mock_get_rules, clean_coco_data: COCOData, caplog
    ):
        """
        GIVEN a set of rules where one rule unexpectedly fails with an exception
        WHEN `run_scan` is called
        THEN the scan should complete and report the failure as a finding
        AND still include results from successful rules.
        """
        caplog.set_level(logging.INFO)

        # Arrange: Create mock rules with different behaviors
        def failing_rule(data):
            raise ValueError("This rule has a bug!")

        failing_rule.__name__ = "check_failing_rule"

        def working_rule(data):
            return ["This is a finding from a working rule."]

        working_rule.__name__ = "check_working_rule"

        mock_get_rules.return_value = [failing_rule, working_rule]
        mock_parse_coco.return_value = clean_coco_data

        # Act
        result = core.run_scan("dummy/path.json")

        # Assert
        assert result["total_findings"] == 2  # 1 from working rule, 1 from the error
        assert "check_working_rule" in result["findings"]
        assert "check_failing_rule_execution_error" in result["findings"]
        assert (
            "Rule execution failed with an internal error."
            in result["findings"]["check_failing_rule_execution_error"]
        )
        assert (
            "Rule 'check_failing_rule' failed with an unexpected error." in caplog.text
        )


class TestCoreRunScanEdgeCases:
    """
    Tests the `run_scan` function's behavior in edge-case scenarios.
    """

    @patch("labellint.core.rules.get_all_rules")
    @patch("labellint.core.formats.parse_coco")
    def test_with_no_rules_found(
        self, mock_parse_coco, mock_get_rules, clean_coco_data: COCOData, caplog
    ):
        """
        GIVEN the rule discovery process finds no available rules
        WHEN `run_scan` is called
        THEN it should complete successfully with zero findings and log a warning.
        """
        caplog.set_level(logging.INFO)
        mock_get_rules.return_value = []
        mock_parse_coco.return_value = clean_coco_data

        # Act
        result = core.run_scan("dummy/path.json")

        # Assert
        assert result["total_findings"] == 0
        assert not result["findings"]
        mock_get_rules.assert_called_once()
        assert "Discovered 0 rules to execute" in caplog.text
        assert "No rules were found. The scan will have no effect." in caplog.text

    @patch("labellint.core.rules.get_all_rules")
    @patch("labellint.core.formats.parse_coco")
    def test_aggregation_logic(
        self, mock_parse_coco, mock_get_rules, clean_coco_data: COCOData
    ):
        """
        GIVEN a set of mocked rules with varied outputs
        WHEN `run_scan` is called
        THEN it should correctly aggregate findings only from rules that fail.
        """

        # Arrange
        def rule_a(data):
            return ["Finding A1", "Finding A2"]

        rule_a.__name__ = "check_rule_a"

        def rule_b(data):
            return []  # This rule passes

        rule_b.__name__ = "check_rule_b"

        def rule_c(data):
            return ["Finding C1"]

        rule_c.__name__ = "check_rule_c"

        mock_get_rules.return_value = [rule_a, rule_b, rule_c]
        mock_parse_coco.return_value = clean_coco_data

        # Act
        result = core.run_scan("dummy/path.json")

        # Assert
        assert result["total_findings"] == 3
        assert len(result["findings"]) == 2  # Only two rules had findings
        assert "check_rule_a" in result["findings"]
        assert "check_rule_c" in result["findings"]
        assert "check_rule_b" not in result["findings"]  # Rule that passed is not a key
        assert result["findings"]["check_rule_a"] == ["Finding A1", "Finding A2"]


class TestCoreFormatters:
    """
    Covers lines 56-65 in core.py.

    Tests the helper functions responsible for formatting the ScanResult
    into different output formats.
    """

    def test_format_json(self, clean_coco_data: COCOData):
        """
        Ensures the format_json function correctly serializes a ScanResult.
        """
        # Arrange
        scan_result: core.ScanResult = {
            "data": clean_coco_data,
            "findings": {
                "check_rule_one": ["Finding A"],
            },
            "total_findings": 1,
        }

        # Act
        json_string = core.format_json(scan_result)
        data = json.loads(json_string)

        # Assert
        assert "summary" in data
        assert "findings" in data
        assert data["summary"]["total_findings"] == 1
        assert "check_rule_one" in data["findings"]
        assert data["findings"]["check_rule_one"] == ["Finding A"]
