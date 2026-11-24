"""LaTeX cleaning, templating and pdflatex compilation with LLM self-healing.

Usage::

    result = await build_pdf(raw_llm_answer, fixer=llm.fix_latex)
    if result.ok:
        send(result.pdf)
    else:
        send_text(result.latex)

Run ``python pdf_builder.py`` to compile a small self-test document (used in
the Docker build to make sure the TeX installation is complete).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

PDFLATEX_TIMEOUT_S = 60
MAX_FIX_ATTEMPTS = 2
LOG_TAIL_LINES = 40
MAX_BODY_CHARS = 120_000

BODY_PLACEHOLDER = "%%BODY%%"

LATEX_TEMPLATE = r"""\documentclass[12pt,a4paper]{article}
\usepackage{cmap}% searchable / copyable Cyrillic text in the PDF
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english,russian]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{indentfirst}
\usepackage[a4paper,margin=2cm]{geometry}

\theoremstyle{plain}
\newtheorem{theorem}{Теорема}
\newtheorem{lemma}{Лемма}
\newtheorem{corollary}{Следствие}
\theoremstyle{definition}
\newtheorem{definition}{Определение}
\newtheorem{example}{Пример}
\theoremstyle{remark}
\newtheorem{remark}{Замечание}

\setlength{\parskip}{0.4em}
\allowdisplaybreaks

\begin{document}
%%BODY%%
\end{document}
"""

Fixer = Callable[[str, str], Awaitable[str]]


@dataclass(slots=True)
class PdfResult:
    pdf: bytes | None
    latex: str
    error: str | None = None
    fix_attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.pdf is not None


class LatexCompileError(Exception):
    def __init__(self, log_tail: str) -> None:
        super().__init__(log_tail)
        self.log_tail = log_tail


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```[ \t]*(?:latex|tex|LaTeX)?[ \t]*\n?(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PREAMBLE_LINE_RE = re.compile(
    r"^\s*\\(?:documentclass|usepackage|RequirePackage|newtheorem|theoremstyle)\b.*$",
    re.MULTILINE,
)
# Commands that could read/write server files or run code: removed unconditionally.
_DANGEROUS_RE = re.compile(
    r"\\(?:input|include|includeonly|openin|openout|read|readline|write|immediate|"
    r"write18|ShellEscape|directlua|catcode|verbatiminput|lstinputlisting|"
    r"includegraphics|special|InputIfFileExists|IfFileExists)\b(?:\s*\{[^{}]*\})?"
)

_UNICODE_MATH = {
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "≡": r"\equiv",
    "∞": r"\infty", "→": r"\to", "←": r"\leftarrow", "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow", "↔": r"\leftrightarrow", "∈": r"\in", "∉": r"\notin",
    "⊂": r"\subset", "⊆": r"\subseteq", "∪": r"\cup", "∩": r"\cap", "∅": r"\varnothing",
    "∀": r"\forall", "∃": r"\exists", "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
    "∮": r"\oint", "∂": r"\partial", "∇": r"\nabla", "√": r"\sqrt{}", "±": r"\pm",
    "∓": r"\mp", "×": r"\times", "·": r"\cdot", "÷": r"\div", "°": r"^\circ",
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon",
    "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau",
    "φ": r"\varphi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega", "Γ": r"\Gamma",
    "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Π": r"\Pi", "Σ": r"\Sigma",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega", "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}", "⋅": r"\cdot",
}
_UNICODE_PLAIN = {"−": "-", " ": "~", "​": "", "﻿": "", "…": r"\ldots{}"}
# Emoji & pictographs: pdflatex cannot render them.
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def clean_latex(raw: str) -> str:
    """Turn an LLM answer into a safe LaTeX document body."""
    text = _THINK_RE.sub("", raw or "").strip()

    fenced = _FENCE_RE.findall(text)
    if fenced:
        text = "\n\n".join(block.strip() for block in fenced)
    else:
        # Stray single fence markers, e.g. an unclosed ```latex
        text = re.sub(r"^```[a-zA-Z]*\s*$", "", text, flags=re.MULTILINE)

    begin = text.find(r"\begin{document}")
    if begin != -1:
        text = text[begin + len(r"\begin{document}"):]
    end = text.rfind(r"\end{document}")
    if end != -1:
        text = text[:end]

    text = _PREAMBLE_LINE_RE.sub("", text)
    text = _DANGEROUS_RE.sub("", text)

    for src, dst in _UNICODE_PLAIN.items():
        text = text.replace(src, dst)
    for src, dst in _UNICODE_MATH.items():
        text = text.replace(src, rf"\ensuremath{{{dst}}}")
    text = _EMOJI_RE.sub("", text)

    text = text.strip()
    if len(text) > MAX_BODY_CHARS:
        text = text[:MAX_BODY_CHARS]
    return text


def render_document(body: str) -> str:
    return LATEX_TEMPLATE.replace(BODY_PLACEHOLDER, body)


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #


def _sandbox_env() -> dict[str, str]:
    """Minimal environment for pdflatex: no API keys, paranoid file access."""
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE")
        if key in os.environ
    }
    env["openin_any"] = "p"
    env["openout_any"] = "p"
    env["shell_escape"] = "f"
    return env


_ERROR_LINE_RE = re.compile(r"^(?:!|\./main\.tex:(\d+):)")
_BODY_START_LINE = LATEX_TEMPLATE[: LATEX_TEMPLATE.index(BODY_PLACEHOLDER)].count("\n") + 1


def _log_tail(log_path: Path, fallback: str, document: str) -> str:
    """Extract the first error with context and the offending source line."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = fallback.splitlines()

    for idx, line in enumerate(lines):
        match = _ERROR_LINE_RE.match(line)
        if not match:
            continue
        context: list[str] = []
        for ctx_line in lines[idx: idx + 10]:
            if ctx_line.startswith("Here is how much of TeX's memory"):
                break
            context.append(ctx_line)
        text = "\n".join(context).strip()
        lineno = match.group(1)
        if lineno:
            doc_lines = document.splitlines()
            n = int(lineno)
            if 1 <= n <= len(doc_lines):
                body_line = n - _BODY_START_LINE + 1
                text += f"\nСтрока {body_line} фрагмента: {doc_lines[n - 1].strip()}"
        return text
    return "\n".join(lines[-LOG_TAIL_LINES:])


def _compile_sync(document: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="mathbot_") as tmp:
        workdir = Path(tmp)
        (workdir / "main.tex").write_text(document, encoding="utf-8")
        try:
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-no-shell-escape",
                    "-file-line-error",
                    "main.tex",
                ],
                cwd=workdir,
                env=_sandbox_env(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=PDFLATEX_TIMEOUT_S,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
            raise LatexCompileError(_log_tail(workdir / "main.log", stdout, document)) from exc
        except subprocess.TimeoutExpired as exc:
            raise LatexCompileError(
                f"pdflatex timed out after {PDFLATEX_TIMEOUT_S}s (possible infinite loop)."
            ) from exc

        pdf_path = workdir / "main.pdf"
        if not pdf_path.exists():
            raise LatexCompileError(_log_tail(workdir / "main.log", "PDF was not produced.", document))
        return pdf_path.read_bytes()


async def compile_latex(body: str) -> bytes:
    """Compile a document body to PDF bytes (runs pdflatex in a worker thread)."""
    return await asyncio.to_thread(_compile_sync, render_document(body))


async def build_pdf(raw_answer: str, fixer: Fixer | None = None, max_fix_attempts: int = MAX_FIX_ATTEMPTS) -> PdfResult:
    """Clean, compile and, on failure, ask the LLM to repair LaTeX (self-healing)."""
    body = clean_latex(raw_answer)
    if not body:
        return PdfResult(pdf=None, latex=raw_answer, error="empty LaTeX body")

    attempt = 0
    while True:
        try:
            pdf = await compile_latex(body)
            if attempt:
                logger.info("PDF compiled after %d self-healing attempt(s)", attempt)
            return PdfResult(pdf=pdf, latex=body, fix_attempts=attempt)
        except FileNotFoundError:
            logger.error("pdflatex is not installed — sending text instead")
            return PdfResult(pdf=None, latex=body, error="pdflatex not found", fix_attempts=attempt)
        except LatexCompileError as exc:
            logger.warning("pdflatex failed (attempt %d): %s", attempt + 1, exc.log_tail[:500])
            if fixer is None or attempt >= max_fix_attempts:
                return PdfResult(pdf=None, latex=body, error=exc.log_tail, fix_attempts=attempt)
            attempt += 1
            try:
                fixed = clean_latex(await fixer(body, exc.log_tail))
            except Exception as fix_exc:  # noqa: BLE001
                logger.error("LaTeX fixer failed: %s", fix_exc)
                return PdfResult(pdf=None, latex=body, error=exc.log_tail, fix_attempts=attempt)
            if fixed:
                body = fixed


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

_SELFTEST_BODY = r"""
\section*{Самопроверка}
Проверка кириллицы и формул: $\int_0^1 x^2\,dx = \frac{1}{3}$, π ≈ 3.14.
\begin{definition}Предел функции $f$ в точке $a$ равен $L$, если
\[ \forall \varepsilon>0\ \exists \delta>0:\ 0<|x-a|<\delta \Rightarrow |f(x)-L|<\varepsilon. \]
\end{definition}
\begin{theorem}$\displaystyle\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$.\end{theorem}
\begin{proof}Индукция по $n$.\end{proof}
\begin{align*} \boxed{x} &= \mathbb{R} \setminus \{0\} \end{align*}
"""


def _selftest() -> int:
    logging.basicConfig(level=logging.INFO)
    pdf = _compile_sync(render_document(clean_latex(_SELFTEST_BODY)))
    print(f"pdflatex self-test OK, {len(pdf)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
