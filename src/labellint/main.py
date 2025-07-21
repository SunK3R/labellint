# ==============================================================================
# labellint.main: The Command-Line Interface
#
# This module defines the user-facing command-line interface for `labellint`
# using the Typer framework. It is the sole entry point for user interaction.
#
# Its responsibilities are:
#   1. Define CLI commands, arguments, and options.
#   2. Orchestrate calls to the backend `core` and `rules` modules.
#   3. Handle all terminal input and output (IO).
#   4. Present results and errors to the user in a beautiful, structured
#      format using the `rich` library.
#
# This layer contains ZERO business logic related to parsing or rule execution.
# It is a pure presentation and control layer.
# ==============================================================================

import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback
from rich.tree import Tree

# Attempt to import from the package structure. This allows the CLI to work
# correctly when installed as a package. A guard is in place for developers
# who might accidentally run this file as a script.
try:
    from . import __version__, core, rules
    from .formats import ParseError
except ImportError:
    print(
        "Fatal Error: `labellint` could not be run.\n"
        "This file is part of a package and cannot be executed directly.\n"
        "Please install `labellint` properly, e.g., `pip install -e .`"
    )
    sys.exit(1)


# ==============================================================================
# Initialization and Configuration
# ==============================================================================

# Use a custom theme for consistent and professional-looking output.
custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "title": "bold magenta",
        "path": "bold cyan",
        "rule": "bold yellow",
        "finding_count": "bold red",
        "dim": "dim",
    }
)

# A single, themed Console instance for all terminal output.
console = Console(theme=custom_theme)

# Rich's traceback handler provides clean, readable stack traces on error.
# We disable showing local variables for security and simplicity.
install_rich_traceback(show_locals=False, extra_lines=1, console=console)

# The main Typer application object.
app = typer.Typer(
    name="labellint",
    help="A high-precision CLI for finding errors in your CV annotation files.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ==============================================================================
# Private Helper Functions for Report Rendering
# ==============================================================================


def _format_rule_name(name: str) -> str:
    """Converts a function name like 'check_case_consistency' to 'Case Consistency'."""
    return name.replace("check_", "").replace("_", " ").title()


def _print_report(result: core.ScanResult) -> None:
    """
    Renders the entire scan report to the console from a ScanResult object.
    This is the single source of truth for report presentation.
    """
    data = result["data"]
    findings = result["findings"]
    total_findings = result["total_findings"]

    # --- Build and Print Summary Panel ---
    status_style = "error" if total_findings > 0 else "success"
    status_text = f"[{status_style}]{total_findings:,} issues found[/{status_style}]"

    summary_table = Table.grid(expand=True, padding=(0, 2))
    summary_table.add_column(style="info", no_wrap=True)
    summary_table.add_column()
    summary_table.add_row("Images:", f"{len(data.images):,}")
    summary_table.add_row("Annotations:", f"{len(data.annotations):,}")
    summary_table.add_row("Categories:", f"{len(data.categories):,}")
    summary_table.add_row("Scan Result:", status_text)

    console.print(
        Panel(
            summary_table,
            title="[title]Scan Summary[/title]",
            border_style="info",
            padding=(1, 2),
        )
    )

    # --- Build and Print Findings Tree (if any) ---
    if total_findings > 0:
        tree = Tree("[error]Detailed Findings[/error]", guide_style="bold bright_black")
        sorted_results = sorted(findings.items(), key=lambda item: item[0])

        for rule_name, rule_findings in sorted_results:
            rule_func = getattr(rules, rule_name)
            docstring = (rule_func.__doc__ or "No description.").strip().split("\n")[0]
            num_rule_findings = len(rule_findings)

            rule_node = tree.add(
                f"[rule]{_format_rule_name(rule_name)}[/rule] "
                f"([finding_count]{num_rule_findings}[/finding_count])\n"
                f"[dim]{docstring}[/dim]"
            )

            for i, finding_text in enumerate(rule_findings):
                if i >= 10:  # Truncate long lists of findings for readability
                    rule_node.add(f"[dim]... and {num_rule_findings - 10} more.[/dim]")
                    break
                escaped_finding = escape(finding_text)
                rule_node.add(Text.from_markup(f"• [white]{escaped_finding}[/white]"))
        console.print(tree)
    else:
        console.print(
            Panel(
                "✅ [success]No issues found.[/success] Your annotations look clean!",
                title="[title]Result[/title]",
                border_style="success",
                padding=(1, 2),
            )
        )


def _version_callback(value: bool) -> None:
    """Callback function to display the version and exit."""
    if value:
        console.print(f"labellint version: [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


# ==============================================================================
# CLI Commands
# ==============================================================================


@app.command(name="scan", help="Scan a single annotation file for anomalies.")
def scan(
    filepath: Annotated[
        Path,
        typer.Argument(
            help="Path to the annotation file (e.g., coco.json).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--out",
            "-o",
            help="Path to save the full, untruncated report. (e.g., report.json)",
            writable=True,
            resolve_path=True,
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Format for the output report file.",
            case_sensitive=False,
        ),
    ] = "json",
) -> None:
    """
    Orchestrates the annotation scanning process and handles all user-facing IO.
    """
    console.print(
        Panel(
            f"[info]Starting Scan[/info]\n"
            f"   File: [title]{filepath.name}[/title]\n"
            f"   Path: [path]{filepath.parent}[/path]",
            title="[bold]LabelLint[/bold]",
            subtitle="[dim]The Quality Gate for Annotations[/dim]",
            expand=False,
            border_style="info",
        )
    )

    start_time = time.perf_counter()
    result: Optional[core.ScanResult] = None

    try:
        with console.status(
            "[bold green]Analyzing annotations...[/bold green]", spinner="dots"
        ):
            # Delegate all logic to the core engine. This layer only handles IO.
            # The core engine returns structured data, never a formatted string.
            result = core.run_scan(filepath=str(filepath))

        # Default behavior: print rich, truncated report to console.
        _print_report(result)

        # --- Output Handling Logic ---
        if output_file:
            # User wants a file-based report.
            if output_format not in core.SUPPORTED_FORMATS:
                console.print(
                    f"[error]Unsupported format '{output_format}'. "
                    f"Supported formats: {list(core.SUPPORTED_FORMATS.keys())}[/error]"
                )
                raise typer.Exit(code=1)

            console.print(
                f"\n [info][bold]Formatting report as '{output_format}'...[/bold][info]"
            )
            formatter = core.SUPPORTED_FORMATS[output_format]
            report_content = formatter(result)

            output_file.write_text(report_content, encoding="utf-8")
            console.print(
                Panel(
                    f"✅ [success]Full report saved to:[/success]\n[path]{output_file}[/path]",
                    title="[title]Report Saved[/title]",
                    border_style="success",
                )
            )

    except ParseError as e:
        console.print(
            Panel(
                f"[error]Parsing Failed[/error]\n\nCould not read or validate the annotation file:\n[yellow]{e}[/yellow]",
                title="[error]Error[/error]",
                border_style="error",
            )
        )
        raise typer.Exit(code=1) from e
    except Exception:
        console.print(
            Panel(
                "[error]An unexpected critical error occurred.[/error]\nSee traceback below for details.",
                title="[error]Critical Error[/error]",
                border_style="error",
            )
        )
        raise  # Re-raise for rich traceback handler to display it.
    finally:
        duration = time.perf_counter() - start_time
        console.print(
            f"[dim]Scan completed in {duration:.3f} seconds.[/dim]", justify="right"
        )

    if result and result["total_findings"] > 0:
        raise typer.Exit(code=1)
    else:
        raise typer.Exit(code=0)


@app.command(name="rules", help="List all available linting rules.")
def list_rules() -> None:
    """Displays a formatted list of all discoverable linting rules."""
    console.print(
        Panel(
            "[title]Available Linting Rules[/title]",
            border_style="info",
            padding=(1, 2),
        )
    )
    try:
        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column("Rule Name", style="rule")
        table.add_column("Description", style="dim")

        available_rules = rules.get_all_rules()
        if not available_rules:
            console.print("[warning]No rules found.[/warning]")
            return

        for rule_func in available_rules:
            doc = rule_func.__doc__ or "No description available."
            doc_first_line = doc.strip().split("\n")[0]
            table.add_row(f"• {rule_func.__name__}", doc_first_line)

        console.print(table)

    except Exception as e:
        console.print(f"[error]Error: Could not retrieve rule list: {e}[/error]")
        raise typer.Exit(code=1) from e


@app.callback()
def main_callback(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show the application's version and exit.",
            callback=_version_callback,
            is_eager=True,  # This ensures it runs before any command
        ),
    ] = None,
) -> None:
    """
    Labellint: A high-precision linter for your computer vision annotations.
    """
    pass


if __name__ == "__main__":  # pragma: no cover
    # This block is a safeguard and should not be executed under normal
    # circumstances if the package is installed correctly.
    app()
