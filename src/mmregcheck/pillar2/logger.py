"""Rich console output for verbose TF-BEAR runs."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from rich.console import Console
from rich.panel import Panel


class TFBearLogger:
    """Verbose-only formatting; every method is a no-op when `verbose` is False."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._console = Console() if verbose else None

    def status(self, message: str) -> AbstractContextManager[Any]:
        """Return a spinner context manager when verbose, else a null context."""
        if self._console is None:
            return nullcontext()
        return self._console.status(message, spinner="dots")

    def log_info(self, message: str, style: str = "default") -> None:
        if self._console is None:
            return
        self._console.print(message if style == "default" else f"[{style}]{message}[/{style}]")

    def log_warning(self, message: str) -> None:
        if self._console is not None:
            self._console.print(f"[yellow]Warning: {message}[/yellow]")

    def log_stage(self, stage_num: int, description: str, color: str = "cyan") -> None:
        if self._console is None:
            return
        panel = Panel(
            f"[{color} bold]Stage {stage_num}: {description}[/]",
            title="[bold white]TF-BEAR Pipeline",
            border_style=color,
            padding=(0, 2),
            expand=False,
        )
        self._console.print("\n")
        self._console.print(panel)

    def log_branch(
        self,
        branch_id: str,
        verdict: str,
        delta: float,
        posterior: dict[str, float] | None = None,
    ) -> None:
        """Log the scoring result of one reasoning branch."""
        if self._console is None:
            return
        self._console.rule(f"[bold purple]Branch {branch_id}[/]", style="purple")
        self._console.print(f"[bold]Predicted verdict:[/] {verdict}")
        color = "green" if delta > 0 else "red" if delta < 0 else "default"
        self._console.print(f"[bold]Delta:[/] [{color}]{delta:+.4f}[/{color}]")
        if posterior:
            self._console.print(
                "[bold]Posterior:[/] " + ", ".join(f"{k}: {v:.4f}" for k, v in posterior.items())
            )

    def log_adjudication(self, verdict: str, details: str = "", fallback: bool = False) -> None:
        if self._console is None:
            return
        color = "yellow" if fallback else "bold green"
        title = "Fallback Adjudication" if fallback else "Adjudication Winner"
        content = f"[bold]Verdict:[/] {verdict}" + (f"\n\n[dim]{details}[/dim]" if details else "")
        self._console.print("\n")
        self._console.print(Panel(content, title=f"[{color}]{title}[/]", border_style=color))

    def log_explanation(self, explanation: str) -> None:
        if self._console is None:
            return
        self._console.print("\n")
        self._console.print(
            Panel(
                explanation,
                title="[bold cyan]Final Explanation[/]",
                border_style="cyan",
                padding=(1, 2),
                expand=False,
            )
        )
