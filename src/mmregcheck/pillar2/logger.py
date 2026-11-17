"""Rich console and responsive Jupyter HTML output for verbose TF-BEAR runs."""

from __future__ import annotations

import html
import importlib
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_HTML_COLORS: dict[str, str] = {
    "green": "#2ecc71",
    "bold green": "#2ecc71",
    "bright_blue": "#3b8eea",
    "blue": "#3b8eea",
    "purple": "#c678dd",
    "yellow": "#e5c07b",
    "cyan": "#29b8db",
    "bright_cyan": "#56b6c2",
    "red": "#e06c75",
    "dim": "#858585",
    "default": "var(--vscode-editor-foreground, #d4d4d4)",
}


def _html_panel(
    body_html: str,
    *,
    title: str,
    border_color: str,
    title_color: str | None = None,
    title_align: str = "center",
    padding: str = "6px 12px 10px 12px",
) -> str:
    """Render a 100%-width responsive box with a title embedded in the top border."""
    legend_margin = "0 auto" if title_align == "center" else "0 0 0 8px"
    resolved_title_color = title_color or border_color
    return (
        f'<fieldset style="border: 1px solid {border_color}; border-radius: 5px; '
        f"width: 100%; box-sizing: border-box; min-width: 0; margin: 10px 0 6px 0; "
        f'padding: {padding};">'
        f'<legend style="margin: {legend_margin}; padding: 0 8px; font-weight: 700; '
        f'color: {resolved_title_color};">{html.escape(title)}</legend>'
        f"{body_html}"
        f"</fieldset>"
    )


class TFBearLogger:
    """Verbose-only formatting; every method is a no-op when `verbose` is False."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._is_jupyter = False
        self._html_blocks: list[str] = []
        self._display_handle: Any = None
        if verbose:
            probe = Console()
            self._is_jupyter = probe.is_jupyter
            self._console: Console | None = (
                Console(force_jupyter=False, force_terminal=True, width=88, soft_wrap=False)
                if self._is_jupyter
                else probe
            )
        else:
            self._console = None

    def _emit_html(self, fragment: str) -> bool:
        """Append `fragment` and update the single Jupyter display block in place."""
        self._html_blocks.append(fragment)
        document = (
            '<div style="font-family: var(--vscode-editor-font-family, '
            "'Cascadia Mono', 'Fira Code', Consolas, monospace); "
            "font-size: var(--vscode-editor-font-size, 13px); line-height: 1.45; "
            "color: var(--vscode-editor-foreground, #d4d4d4); "
            'width: 100%; box-sizing: border-box;">' + "".join(self._html_blocks) + "</div>"
        )
        try:
            ipython_display: Any = importlib.import_module("IPython.display")
        except ImportError:
            return False

        html_obj = ipython_display.HTML(document)
        if self._display_handle is None:
            self._display_handle = ipython_display.display(html_obj, display_id=True)
            return self._display_handle is not None
        self._display_handle.update(html_obj)
        return True

    def status(self, message: str) -> AbstractContextManager[Any]:
        """Return a spinner context manager when verbose, else a null context."""
        if self._console is None:
            return nullcontext()
        if self._is_jupyter:
            self.log_info(message, style="dim")
            return nullcontext()
        return self._console.status(message, spinner="dots")

    def log_info(self, message: str, style: str = "default") -> None:
        if self._console is None:
            return
        if self._is_jupyter:
            color = _HTML_COLORS.get(style, _HTML_COLORS["default"])
            if self._emit_html(
                f'<div style="color: {color}; white-space: pre-wrap; '
                f'word-break: break-word; margin: 2px 0;">{html.escape(message)}</div>'
            ):
                return
        self._console.print(message if style == "default" else f"[{style}]{message}[/{style}]")

    def log_prompt(self, prompt: str, title: str = "Prompt") -> None:
        """Log a model prompt in a dimmed panel so it stands apart from responses."""
        if self._console is None:
            return
        if self._is_jupyter:
            body = (
                f'<div style="color: {_HTML_COLORS["dim"]}; white-space: pre-wrap; '
                f'word-break: break-word;">{html.escape(prompt)}</div>'
            )
            if self._emit_html(
                _html_panel(
                    body,
                    title=title,
                    border_color=_HTML_COLORS["dim"],
                    title_align="left",
                )
            ):
                return
        self._console.print(
            Panel(
                Text(prompt, style="dim"),
                title=f"[bold dim]{title}[/]",
                title_align="left",
                border_style="dim",
                padding=(0, 1),
                expand=True,
            )
        )

    def log_response(self, response: str, title: str = "Response") -> None:
        """Log a model response in a highlighted panel so it stands apart from prompts."""
        if self._console is None:
            return
        if self._is_jupyter:
            body = (
                f'<div style="color: {_HTML_COLORS["default"]}; white-space: pre-wrap; '
                f'word-break: break-word;">{html.escape(response)}</div>'
            )
            if self._emit_html(
                _html_panel(
                    body,
                    title=title,
                    border_color=_HTML_COLORS["bright_cyan"],
                    title_align="left",
                )
            ):
                return
        self._console.print(
            Panel(
                Text(response),
                title=f"[bold bright_cyan]{title}[/]",
                title_align="left",
                border_style="bright_cyan",
                padding=(0, 1),
                expand=True,
            )
        )

    def log_warning(self, message: str) -> None:
        if self._console is None:
            return
        if self._is_jupyter and self._emit_html(
            f'<div style="color: {_HTML_COLORS["yellow"]}; white-space: pre-wrap; '
            f'word-break: break-word; margin: 2px 0;">Warning: {html.escape(message)}</div>'
        ):
            return
        self._console.print(f"[yellow]Warning: {message}[/yellow]")

    def log_stage(self, stage_num: int, description: str, color: str = "cyan") -> None:
        if self._console is None:
            return
        if self._is_jupyter:
            if stage_num == 1:
                self._html_blocks = []
                self._display_handle = None
            hex_color = _HTML_COLORS.get(color, _HTML_COLORS["cyan"])
            body = (
                f'<div style="text-align: center; font-weight: 700; color: {hex_color};">'
                f"Stage {stage_num}: {html.escape(description)}</div>"
            )
            if self._emit_html(
                _html_panel(
                    body,
                    title="TF-BEAR Pipeline",
                    border_color=hex_color,
                    title_color="var(--vscode-editor-foreground, #ffffff)",
                    title_align="center",
                )
            ):
                return
        panel = Panel(
            Align.center(f"[{color} bold]Stage {stage_num}: {description}[/]"),
            title="[bold white]TF-BEAR Pipeline",
            border_style=color,
            padding=(0, 2),
            expand=True,
        )
        self._console.print()
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
        if self._is_jupyter:
            delta_color = (
                _HTML_COLORS["green"]
                if delta > 0
                else _HTML_COLORS["red"]
                if delta < 0
                else _HTML_COLORS["default"]
            )
            posterior_line = ""
            if posterior:
                pairs = ", ".join(
                    f'{html.escape(k)}: <span style="color: {_HTML_COLORS["cyan"]};">{v:.4f}</span>'
                    for k, v in posterior.items()
                )
                posterior_line = f"<div><strong>Posterior:</strong> {pairs}</div>"
            branch_html = (
                '<div style="display: flex; align-items: center; width: 100%; margin: 8px 0 4px 0;">'
                f'<span style="flex: 1; border-bottom: 1px solid {_HTML_COLORS["purple"]};"></span>'
                f'<span style="padding: 0 10px; font-weight: 700; color: {_HTML_COLORS["purple"]};">'
                f"Branch {html.escape(branch_id)}</span>"
                f'<span style="flex: 1; border-bottom: 1px solid {_HTML_COLORS["purple"]};"></span>'
                "</div>"
                f"<div><strong>Predicted verdict:</strong> {html.escape(verdict)}</div>"
                f'<div><strong>Delta:</strong> <span style="color: {delta_color};">{delta:+.4f}</span></div>'
                f"{posterior_line}"
            )
            if self._emit_html(branch_html):
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
        if self._is_jupyter:
            details_html = (
                f'<div style="color: {_HTML_COLORS["dim"]}; margin-top: 6px; '
                f'white-space: pre-wrap; word-break: break-word;">{html.escape(details)}</div>'
                if details
                else ""
            )
            body = f"<div><strong>Verdict:</strong> {html.escape(verdict)}</div>{details_html}"
            if self._emit_html(
                _html_panel(
                    body,
                    title=title,
                    border_color=_HTML_COLORS[color],
                    title_align="center",
                )
            ):
                return
        content = f"[bold]Verdict:[/] {verdict}" + (f"\n\n[dim]{details}[/dim]" if details else "")
        self._console.print()
        self._console.print(Panel(content, title=f"[{color}]{title}[/]", border_style=color))

    def log_explanation(self, explanation: str) -> None:
        if self._console is None:
            return
        if self._is_jupyter:
            body = (
                f'<div style="white-space: pre-wrap; word-break: break-word;">'
                f"{html.escape(explanation)}</div>"
            )
            if self._emit_html(
                _html_panel(
                    body,
                    title="Final Explanation",
                    border_color=_HTML_COLORS["cyan"],
                    title_align="center",
                    padding="10px 16px 14px 16px",
                )
            ):
                return
        self._console.print()
        self._console.print(
            Panel(
                explanation,
                title="[bold cyan]Final Explanation[/]",
                border_style="cyan",
                padding=(1, 2),
                expand=True,
            )
        )
