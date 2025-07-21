# ==============================================================================
# tests.test_main: Integration Tests for the Command-Line Interface
#
# This module tests the user-facing CLI defined in `labellint.main`. It uses
# Typer's `CliRunner` to invoke commands programmatically and assert on their
# output and exit codes. This provides high-confidence testing of the entire
# user interaction layer.
#
# Test Design Philosophy:
#   - Black Box Testing: The CLI (`app`) is treated as a complete system.
#     Tests simulate user input and inspect the resulting output (stdout,
#     stderr) and exit codes.
#   - Mocked Backend: The `core.run_scan` function is mocked. This isolates
#     the tests to *only* the CLI's presentation and error-handling logic,
#     preventing them from becoming slow, full end-to-end tests.
#   - Focus on User Experience: Tests validate that error messages are helpful,
#     success messages are clear, and reports are formatted as expected.
# ==============================================================================

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from labellint import main
from labellint.formats import COCOData, ParseError

# We must import `app` from the module that defines it.
from labellint.main import app
from typer.testing import CliRunner

# The CliRunner is the primary tool for testing Typer applications.
runner = CliRunner()

# All tests in this file implicitly use fixtures from `tests/conftest.py`,
# especially `clean_coco_data` and `create_coco_file`.


class TestMainModuleGuards:
    def test_import_error_guard(self):
        """
        GIVEN labellint.main is executed in an environment where its dependencies are missing
        WHEN the module is imported
        THEN it should print a fatal error and exit with code 1.
        """
        # To ensure the import fails, we can temporarily rename a dependency's folder
        # or use a more complex setup. A simpler way is to check the output
        # when a known import will fail. We can simulate this by manipulating the path.
        # This is complex, so we will test the *effect* of the code.

        # A more direct way to test the block is to force the import to fail inside a test
        # This is a functional equivalent of running the script in a broken environment
        script_path = Path(__file__).parent.parent / "src" / "labellint" / "main.py"

        # The command for the subprocess: execute main.py directly
        process = subprocess.run(
            [sys.executable, str(script_path)], capture_output=True, text=True
        )

        # Assert that the custom error message from the except block is printed
        assert process.returncode == 1
        assert "Fatal Error" in process.stdout
        assert "cannot be executed directly" in process.stdout


# By patching at the class level, all methods within the class will use the mock.
@patch("labellint.main.core.run_scan")
class TestScanCommand:
    """A comprehensive test suite for the `labellint scan` command."""

    def test_scan_with_findings(
        self, mock_run_scan: MagicMock, create_coco_file, clean_coco_data: COCOData
    ):
        """
        GIVEN a valid file path and a mocked core engine that returns findings
        WHEN the `scan` command is run
        THEN it should print a full report, include finding details, and exit with code 1.
        """
        # Arrange
        mock_run_scan.return_value = {
            "data": clean_coco_data,
            "findings": {"check_category_case_consistency": ["'Car' vs 'car'"]},
            "total_findings": 1,
        }
        test_file = create_coco_file(clean_coco_data)

        # Act
        result = runner.invoke(app, ["scan", str(test_file)])

        # Assert
        assert (
            result.exit_code == 1
        ), "Should exit with non-zero code when findings exist."
        assert "Scan Summary" in result.stdout
        assert "1 issues found" in result.stdout
        assert "Detailed Findings" in result.stdout
        assert "Case Consistency" in result.stdout
        assert "'Car' vs 'car'" in result.stdout
        mock_run_scan.assert_called_once_with(filepath=str(test_file))

    def test_scan_with_no_findings(
        self, mock_run_scan: MagicMock, create_coco_file, clean_coco_data: COCOData
    ):
        """
        GIVEN a valid file path and a mocked core engine that returns zero findings
        WHEN the `scan` command is run
        THEN it should print a success message and exit with code 0.
        """
        # Arrange
        mock_run_scan.return_value = {
            "data": clean_coco_data,
            "findings": {},
            "total_findings": 0,
        }
        test_file = create_coco_file(clean_coco_data)

        # Act
        result = runner.invoke(app, ["scan", str(test_file)])

        # Assert
        assert result.exit_code == 0
        assert "Scan Summary" in result.stdout
        assert "0 issues found" in result.stdout
        assert "✅ No issues found." in result.stdout
        assert "Detailed Findings" not in result.stdout

    def test_scan_handles_parse_error(
        self, mock_run_scan: MagicMock, create_coco_file, clean_coco_data: COCOData
    ):
        """
        GIVEN the core engine raises a `ParseError`
        WHEN the `scan` command is run
        THEN it should print a formatted parsing error message and exit with code 1.
        """
        # Arrange
        error_message = "This is a test parse error."
        mock_run_scan.side_effect = ParseError(error_message)
        test_file = create_coco_file(clean_coco_data)

        # Act
        result = runner.invoke(app, ["scan", str(test_file)])

        # Assert
        assert result.exit_code == 1
        assert "Parsing Failed" in result.stdout
        assert error_message in result.stdout
        assert "Detailed Findings" not in result.stdout

    def test_scan_handles_unexpected_exception(
        self, mock_run_scan: MagicMock, create_coco_file, clean_coco_data: COCOData
    ):
        """
        GIVEN the core engine raises an unexpected `Exception`
        WHEN the `scan` command is run
        THEN it should print a critical error message and re-raise the exception for the traceback handler.
        """
        # Arrange
        error_message = "A critical unexpected failure."
        mock_run_scan.side_effect = ValueError(error_message)
        test_file = create_coco_file(clean_coco_data)

        # Act
        result = runner.invoke(app, ["scan", str(test_file)])

        # Assert
        assert result.exit_code == 1
        assert "Critical Error" in result.stdout
        assert "An unexpected critical error occurred." in result.stdout
        # Assert that the original exception was allowed to propagate for the rich traceback handler
        assert isinstance(result.exception, ValueError)

    def test_scan_file_not_found(self, mock_run_scan: MagicMock):
        """
        GIVEN a path to a non-existent file
        WHEN the `scan` command is run
        THEN Typer's built-in validation should handle it and exit with code 2.
        """
        # Arrange
        non_existent_path = "no/such/file.json"

        # Act
        result = runner.invoke(app, ["scan", non_existent_path], catch_exceptions=True)

        # Assert
        assert result.exit_code == 2  # Typer's exit code for bad parameters
        assert "does not exist" in result.output

    def test_scan_truncates_long_finding_lists(
        self, mock_run_scan: MagicMock, create_coco_file, clean_coco_data: COCOData
    ):
        """
        GIVEN a rule returns more than 10 findings
        WHEN the `scan` command is run
        THEN the output should be truncated with a '... and X more' message.
        """
        # Arrange
        long_finding_list = [f"Finding #{i}" for i in range(15)]

        mock_run_scan.return_value = {
            "data": clean_coco_data,
            "findings": {"check_category_duplicate_ids": long_finding_list},
            "total_findings": 15,
        }
        test_file = create_coco_file(clean_coco_data)

        # Act
        result = runner.invoke(app, ["scan", str(test_file)])

        # Assert
        assert result.exit_code == 1
        assert "Finding #9" in result.stdout
        assert "Finding #10" not in result.stdout
        assert "... and 5 more." in result.stdout

    def test_scan_with_unsupported_output_format(
        self, mock_run_scan: MagicMock, create_coco_file, clean_coco_data: COCOData
    ):
        """
        Covers lines 236-250 in main.py.

        Ensures the CLI exits gracefully with an error message when an
        unsupported output format is requested.
        """
        # Arrange
        mock_run_scan.return_value = {
            "data": clean_coco_data,
            "findings": {},
            "total_findings": 0,
        }
        test_file = create_coco_file(clean_coco_data)

        # Act
        result = runner.invoke(
            app, ["scan", str(test_file), "--out", "report.txt", "--format", "xml"]
        )

        # Assert
        assert result.exit_code == 1
        assert "Unsupported format 'xml'" in result.stdout

    def test_scan_saves_full_report_to_file(
        self,
        mock_run_scan: MagicMock,
        create_coco_file,
        clean_coco_data: COCOData,
        tmp_path: Path,
    ):
        """
        Covers lines 243-256 in main.py.

        Ensures that when a valid output file is requested, the application correctly
        formats the result, writes it to the specified file, and prints a
        confirmation message to the console.
        """
        # Arrange
        scan_result = {
            "data": clean_coco_data,
            "findings": {"check_category_duplicate_ids": ["Finding 1"]},
            "total_findings": 1,
        }
        mock_run_scan.return_value = scan_result

        input_file = create_coco_file(clean_coco_data)
        output_file = tmp_path / "report.json"

        # Act
        result = runner.invoke(
            app,
            ["scan", str(input_file), "--out", str(output_file), "--format", "json"],
        )

        # Assert: Check CLI behavior
        assert result.exit_code == 1  # Exits with 1 because findings were found
        assert "Formatting report as 'json'" in result.stdout
        assert "Full report saved to" in result.stdout

        # *** FIX: Assert on the filename, which is more robust than the full absolute path. ***
        assert output_file.name in result.stdout

        # Assert: Check file system behavior and content
        assert output_file.exists()
        report_content = json.loads(output_file.read_text())
        assert report_content["summary"]["total_findings"] == 1
        assert "check_category_duplicate_ids" in report_content["findings"]


@patch("labellint.main.rules.get_all_rules")
class TestRulesCommand:
    """A test suite for the `labellint rules` command."""

    def test_rules_success(self, mock_get_all_rules: MagicMock):
        """
        GIVEN the rule engine returns a list of rule functions
        WHEN the `rules` command is run
        THEN it should print a formatted table of rule names and docstrings.
        """

        # Arrange
        def rule_one():
            """This is rule one."""

        def rule_two():
            """This is rule two."""

        mock_get_all_rules.return_value = [rule_one, rule_two]

        # Act
        result = runner.invoke(app, ["rules"])

        # Assert
        assert result.exit_code == 0
        assert "Available Linting Rules" in result.stdout
        assert "rule_one" in result.stdout
        assert "This is rule one" in result.stdout
        assert "rule_two" in result.stdout
        assert "This is rule two" in result.stdout
        mock_get_all_rules.assert_called_once()

    def test_rules_no_rules_found(self, mock_get_all_rules: MagicMock):
        """
        GIVEN the rule engine finds no rules
        WHEN the `rules` command is run
        THEN it should print a "No rules found" message.
        """
        # Arrange
        mock_get_all_rules.return_value = []

        # Act
        result = runner.invoke(app, ["rules"])

        # Assert
        assert result.exit_code == 0
        assert "No rules found" in result.stdout

    def test_rules_handles_exception(self, mock_get_all_rules: MagicMock):
        """
        GIVEN the rule engine raises an exception
        WHEN the `rules` command is run
        THEN it should print a formatted error message and exit with code 1.
        """
        # Arrange
        mock_get_all_rules.side_effect = Exception("Failed to load rules")

        # Act
        result = runner.invoke(app, ["rules"])

        # Assert
        assert result.exit_code == 1
        assert "Error: Could not retrieve rule list" in result.stdout


class TestGeneralCliBehavior:
    def test_invoked_with_no_args(self):
        """WHEN `labellint` is run with no arguments, THEN it should print help and exit with a usage error code."""
        result = runner.invoke(app, [], env={"NO_COLOR": "1", "TERM": "dumb"})
        assert result.exit_code in [0, 2]
        assert "Usage: labellint [OPTIONS] COMMAND [ARGS]..." in result.stdout

    def test_help_option(self):
        """WHEN `labellint --help` is run, THEN it should print the help message and exit successfully."""
        env = {"NO_COLOR": "1", "TERM": "dumb"}
        result = runner.invoke(app, ["--help"], env=env)
        assert result.exit_code == 0
        assert "Usage: labellint [OPTIONS] COMMAND [ARGS]..." in result.stdout

    def test_version_option(self):
        """
        WHEN `labellint --version` is run
        THEN it should print the version and exit successfully.
        """
        with patch("labellint.main.__version__", "1.2.3"):
            result = runner.invoke(app, ["--version"])
            assert result.exit_code == 0
            assert "labellint version: 1.2.3" in result.stdout

    @patch("labellint.core.SUPPORTED_FORMATS", {})
    def test_scan_with_invalid_format_and_no_supported_formats(
        self: MagicMock, create_coco_file, clean_coco_data: COCOData
    ):
        """
        Covers lines 243-250 in main.py.
        """
        # This test function must be at the module level to avoid conflicts with class-level patches.
        # We also need to patch `run_scan` for this specific test.
        with patch("labellint.main.core.run_scan") as mock_run_scan_inner:
            mock_run_scan_inner.return_value = {
                "data": clean_coco_data,
                "findings": {},
                "total_findings": 0,
            }
            test_file = create_coco_file(clean_coco_data)

            result = runner.invoke(
                app, ["scan", str(test_file), "--out", "report.txt", "--format", "json"]
            )

            assert result.exit_code == 1
            assert "Unsupported format 'json'" in result.stdout
            assert "Supported formats: []" in result.stdout

    def test_import_error_guard_is_covered(self):
        """
        Covers lines 39-45 in main.py.

        This test simulates an ImportError within the test runner's process by
        temporarily removing a key dependency from Python's module cache and
        then forcing main.py to be reloaded. This is an advanced technique
        required to get coverage on module-level import guards.
        """
        # Define the modules that `main` imports relatively
        modules_to_remove = [
            "labellint.core",
            "labellint.rules",
            "labellint.formats",
            "labellint.__version__",
        ]

        # Create a dictionary where the modules are set to None
        patched_modules = {mod: None for mod in modules_to_remove}

        # Patch the system's module cache. Inside this block, Python will think
        # these modules have not been imported.
        with patch.dict("sys.modules", patched_modules):
            # The 'except ImportError' block in main.py calls sys.exit(1),
            # which raises a SystemExit exception. We must catch it.
            with pytest.raises(SystemExit) as exc_info:
                # importlib.reload forces Python to re-execute the module's code
                importlib.reload(main)

        # Assert that the script tried to exit with the correct code,
        # proving the `except` block was executed.
        assert exc_info.value.code == 1
