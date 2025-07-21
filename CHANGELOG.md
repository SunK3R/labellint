# Changelog

All notable changes to the `labellint` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

This section documents changes that have been merged into the main branch but have not yet been part of a formal release.

### Added
-   Initial planning for Pascal VOC and YOLO format parsers in `labellint.formats`.
-   Scaffolding for a Markdown (`--format md`) report generator.

### Changed
-   (No changes yet)

### Fixed
-   (No changes yet)

---

## [0.1.0] - 2025-07-21

This is the initial public release of `labellint`. This version establishes the core functionality of the command-line tool, including the linter engine, rule system, and support for the COCO annotation format.

### Added

#### **Core Application & CLI**
-   Introduced the primary `labellint` command-line application using `Typer` and `rich` for a robust and user-friendly interface.
-   **`scan` command:** The main entry point for analyzing annotation files. It provides a detailed, color-coded summary report directly to the terminal.
-   **`rules` command:** A utility to list all available validation rules and their descriptions.
-   **File Export:** Implemented `--out` and `--format` options on the `scan` command to save full, untruncated reports in machine-readable JSON format.
-   **CI/CD Integration:** The `scan` command now exits with a status code of `0` on success (no findings) and `1` on failure (one or more findings), enabling its use as a quality gate in automated workflows.

#### **Data Parsing & Validation**
-   Implemented a high-performance parser for the **COCO Object Detection** annotation format.
-   Established a strict data validation layer using `Pydantic` models to ensure that the linter engine only operates on schema-compliant, type-safe data.

#### **Rule Engine & Built-in Rules**
-   Established a discoverable rule engine that automatically identifies and executes any function prefixed with `check_` in the `rules.py` module.
-   Implemented a comprehensive suite of 11 built-in validation rules, categorized as follows:
    -   **Schema & Relational Integrity:**
        -   `check_category_case_consistency`
        -   `check_category_duplicate_ids`
        -   `check_category_duplicate_names`
        -   `check_relation_unmatched_annotations` (orphaned annotations)
        -   `check_relation_unmatched_category`
        -   `check_relation_images_without_annotations`
    -   **Geometric & Attribute Consistency:**
        -   `check_geometry_zero_area_bboxes`
        -   `check_geometry_bbox_out_of_bounds`
        -   `check_attribute_area_bbox_mismatch`
    -   **Statistical Anomaly Detection:**
        -   `check_statistical_bbox_aspect_ratio_outliers` (using IQR)
        -   `check_statistical_class_distribution_imbalance`

#### **Project Infrastructure & Quality**
-   Established a professional project structure using the `src` layout.
-   Configured a full suite of development quality tools:
    -   **Testing:** `pytest` and `pytest-cov`, achieving 100% test coverage for all modules.
    -   **Linting & Formatting:** `ruff`, configured with a strict ruleset.
    -   **Static Type Checking:** `mypy`, configured in strict mode.
-   Created a comprehensive `README.md` including installation, usage, and development guidelines.
-   Added a detailed `.gitignore` file to ensure a clean repository.

### Fixed

-   **Rule Logic:** Corrected the `check_attribute_area_bbox_mismatch` rule to correctly ignore annotations with complex polygon segmentation data, eliminating a major source of false positives on standard COCO datasets.
-   **Rule Robustness:** Hardened the `check_statistical_class_distribution_imbalance` rule against `AttributeError` when encountering annotations with invalid category IDs. Also added a guard clause to prevent it from running on small datasets where its results would not be meaningful.
-   **CLI Behavior:** Resolved an issue where the CLI would incorrectly display a "Critical Error" message when exiting with findings. The non-zero exit for CI/CD is now handled as part of the normal program flow.
-   **Packaging:** Corrected the `pyproject.toml` configuration and project structure to resolve a `ModuleNotFoundError` upon installation, ensuring the `labellint` command is correctly installed and placed on the user's `PATH`.
-   **Static Analysis:** Resolved all reported issues from `mypy` (type inconsistencies) and `ruff` (style violations, unused code, and best-practice deviations) to ensure full compliance with the project's quality standards.
-   **Testing:** Corrected multiple fragile and failing tests, particularly those related to module-level guards and command-line argument parsing, to ensure the test suite is 100% reliable.

[Unreleased]: https://github.com/SunK3R/labellint/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SunK3R/labellint/releases/tag/v0.1.0