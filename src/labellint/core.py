# ==============================================================================
# labellint.core: The Linter Engine
#
# This module orchestrates the entire scanning process. It is the logical
# heart of the application, responsible for coordinating the data parsing and
# rule execution flow.
#
# Architectural Principles:
#   - Decoupling: This module is completely decoupled from the presentation
#     layer (`main.py`). It does not know about Typer, Rich, or how the
#     results will be displayed. It only deals with data.
#   - Data-Centric: Its input is a file path, and its output is a structured
#     dictionary (`ScanResult`). It does not produce formatted strings.
#   - Orchestration, Not Implementation: The core logic for parsing and
#     validation resides in `formats.py` and `rules.py` respectively. This
#     module's job is to call them in the correct order and aggregate the results.
# ==============================================================================

import json
import logging
from typing import Any, Callable, Dict, List, TypedDict

from . import formats, rules
from .formats import COCOData, ParseError

# --- Setup Internal Logger ---
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Contracts (TypedDicts)
# ==============================================================================


class ScanResult(TypedDict):
    """
    A TypedDict that defines the structured result of a scan.

    This serves as a formal data contract between the core engine and any
    layer that calls it (e.g., the CLI or a future API).
    """

    data: COCOData
    findings: Dict[str, List[str]]
    total_findings: int


# =============================================================================
# Output Formatting Functions
# =============================================================================


def format_json(result: ScanResult) -> str:
    """Formats the full, untruncated scan result as a JSON string."""
    # We don't need the raw data in the JSON output, just the findings.
    output_dict = {
        "summary": {
            "total_findings": result["total_findings"],
            "images_scanned": len(result["data"].images),
            "annotations_scanned": len(result["data"].annotations),
            "categories_found": len(result["data"].categories),
        },
        "findings": result["findings"],  # This is the full, untruncated list.
    }
    return json.dumps(output_dict, indent=2)


# A dictionary to map format names to their respective functions.
# This makes the engine extensible to new formats like Markdown or HTML.
SUPPORTED_FORMATS: Dict[str, Callable[[ScanResult], Any]] = {
    "json": format_json,
}

# ==============================================================================
# The Core Engine Function
# ==============================================================================


def run_scan(filepath: str) -> ScanResult:
    """
    Executes a full scan on an annotation file and returns structured findings.

    This function orchestrates the entire linting pipeline:
    1.  Delegates parsing and validation of the annotation file to `formats.parse_coco`.
    2.  Discovers all available linting rules from the `rules` module.
    3.  Executes each rule against the validated data.
    4.  Aggregates all findings into a structured `ScanResult` dictionary.
    5.  Raises any `ParseError` to be handled by the calling layer.

    Args:
        filepath: The absolute path to the annotation file.

    Returns:
        A `ScanResult` dictionary containing the original data and all findings.

    Raises:
        ParseError: If the file cannot be read or fails schema validation.
    """
    logger.info(f"Core engine starting scan for: {filepath}")

    # --- Stage 1: Parsing and Validation ---
    # We wrap this in a try...except block to clearly delineate the parsing
    # phase. If this fails, the scan is aborted, and the error is propagated
    # to the caller (e.g., the CLI) for user-facing error reporting.
    try:
        data: COCOData = formats.parse_coco(filepath)
        logger.info("Data parsing and validation successful.")
    except ParseError:
        logger.error(
            "Core engine caught a parsing error. Aborting scan.", exc_info=True
        )
        # Re-raise the exception to be handled by the presentation layer.
        raise

    # --- Stage 2: Rule Discovery ---
    all_rules = rules.get_all_rules()
    logger.info(f"Discovered {len(all_rules)} rules to execute.")
    if not all_rules:
        logger.warning("No rules were found. The scan will have no effect.")

    # --- Stage 3: Rule Execution and Aggregation ---
    results: Dict[str, List[str]] = {}
    total_findings = 0
    for rule_func in all_rules:
        rule_name = rule_func.__name__
        try:
            logger.debug(f"Executing rule: {rule_name}")
            findings = rule_func(data)
            if findings:
                num_findings = len(findings)
                results[rule_name] = findings
                total_findings += num_findings
                logger.info(f"Rule '{rule_name}' found {num_findings} issues.")
        except Exception:
            # This is a critical safeguard. If a single rule fails due to an
            # unexpected bug, we log it and continue, ensuring one broken
            # rule doesn't crash the entire scan.
            logger.exception(
                f"Rule '{rule_name}' failed with an unexpected error. Skipping."
            )
            # Optionally, we could add a finding about the failed rule itself.
            error_finding = ["Rule execution failed with an internal error."]
            results[f"{rule_name}_execution_error"] = error_finding
            total_findings += 1

    logger.info(f"Scan complete. Total findings: {total_findings}")

    # --- Stage 4: Return Structured Result ---
    # The final result is returned as a dictionary, fulfilling the `ScanResult`
    # data contract.
    return {
        "data": data,
        "findings": results,
        "total_findings": total_findings,
    }
