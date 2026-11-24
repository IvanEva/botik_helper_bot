"""LLM routing: Google AI Studio (Gemini) as primary, OpenRouter as fallback.

* Gemini handles text and native multimodal input (images, audio, PDF).
* On any Gemini failure (429 / 5xx / network / timeout / empty answer) a
  text-only request is transparently re-routed to OpenRouter.
* ``/or`` forces OpenRouter.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import AsyncOpenAI, OpenAIError

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENROUTER_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "openrouter/free",
]
OPENROUTER_SAFE_DEFAULT = "openrouter/free"  # meta-router, always available
DEFAULT_OPENROUTER_MAX_TOKENS = 24000
# OpenRouter accepts a limited number of entries in the "models" routing list.
OPENROUTER_MAX_MODELS = 3

# Gemini models tried in order after PRIMARY_GEMINI_MODEL (strongest first, lite last).
# Unknown names are dropped at startup by validate_gemini_models().
DEFAULT_GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]
# Total time we are willing to spend walking the Gemini chain before going to OpenRouter.
DEFAULT_GEMINI_TIME_BUDGET_S = 200
# How long a failing model is skipped (seconds), by error kind.
COOLDOWN_OVERLOADED_S = 180   # 5xx / timeout / network
COOLDOWN_RATE_LIMIT_S = 300   # 429 quota
COOLDOWN_NOT_FOUND_S = 6 * 3600  # 404 model does not exist for this key

REQUEST_TIMEOUT_S = 240
LLM_HTTP_STATUS_FALLBACK = {408, 429, 500, 502, 503, 504}


class Mode(str, Enum):
    SOLVE = "solve"  # math problem: deep reasoning
    NOTE = "note"    # structured lecture notes / general questions: fast
    FIX = "fix"      # repair broken LaTeX: fast


@dataclass(slots=True)
class MediaPart:
    data: bytes
    mime_type: str


@dataclass(slots=True)
class LLMResult:
    text: str
    provider: str
    model: str


class LLMError(Exception):
    """Raised when no provider could produce an answer. Message is user-facing."""


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_LATEX_RULES = r"""
ФОРМАТ ВЫВОДА (обязательно):
- Выводи ТОЛЬКО чистый LaTeX-код, который будет вставлен между \begin{document} и \end{document}.
- НЕ пиши преамбулу: никаких \documentclass, \usepackage, \begin{document}, \end{document}.
- НЕ используй Markdown, НЕ оборачивай ответ в ```latex ... ``` и не добавляй пояснений вне LaTeX.
- Доступны только пакеты: amsmath, amssymb, mathtools, amsthm, geometry, graphicx, indentfirst.
- Готовые окружения: theorem, lemma, corollary, definition, example, remark, proof.
- Нельзя: \input, \include, картинки, TikZ, \tableofcontents, \bibliography, пакеты не из списка.
- Текст на русском. Формулы: $...$ внутри строки, \[ ... \] или align*/gather* для выключных.
- Экранируй спецсимволы в тексте: \%, \&, \_, \#, \$.
- Не используй Unicode-символы математики (≤, ∞, α и т.п.) — только команды LaTeX (\leq, \infty, \alpha).
- Заголовки: \section*{...}, \subsection*{...}. Списки: itemize / enumerate.
"""

SYSTEM_PROMPTS: dict[Mode, str] = {
    Mode.SOLVE: (
        "Ты — опытный преподаватель высшей математики. Реши задачу пользователя "
        "строго, подробно и по шагам: матанализ, линейная алгебра, дифференциальные "
        "уравнения, теория вероятностей, комплексный анализ и т.д. Проверь ответ.\n"
        "Структура: \\section*{Условие} (переформулируй задачу), \\section*{Решение} "
        "(пронумерованные шаги с обоснованием), \\section*{Ответ} (итог в рамке "
        "\\boxed{...} внутри формулы).\n"
        "Если на изображении несколько задач — реши все, каждую в своём разделе."
        + _LATEX_RULES
    ),
    Mode.NOTE: (
        "Ты — преподаватель высшей математики, составляющий качественные структурированные "
        "конспекты. По запросу, теме, фото или аудиозаписи лекции составь конспект: "
        "\\section*{} с названием темы, ключевые определения (definition), теоремы "
        "(theorem) с краткими доказательствами или идеями доказательств, формулы, "
        "разобранные примеры (example), типичные ошибки (remark). Если на входе "
        "аудио — сначала пойми его содержание, затем составь конспект (без дословной "
        "расшифровки). Если это просто вопрос — дай чёткий структурированный ответ."
        + _LATEX_RULES
    ),
    Mode.FIX: (
        "Ты — эксперт по LaTeX. Тебе дан фрагмент тела документа и лог ошибки pdflatex. "
        "Исправь ВСЕ синтаксические ошибки, сохранив содержание полностью. "
        "Верни ПОЛНЫЙ исправленный фрагмент целиком."
        + _LATEX_RULES
    ),
}

DEFAULT_USER_PROMPT: dict[Mode, str] = {
    Mode.SOLVE: "Реши задачу (или задачи) из вложения.",
    Mode.NOTE: "Составь подробный структурированный конспект по вложению.",
    Mode.FIX: "",
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Some reasoning models (DeepSeek R1 etc.) leak <think> blocks into content."""
    text = _THINK_RE.sub("", text)
    # Unclosed <think> at the start: drop everything up to the closing tag if any.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class LLMService:
    def __init__(
        self,
        gemini_api_key: str | None,
        openrouter_api_key: str | None,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        openrouter_models: list[str] | None = None,
        openrouter_max_tokens: int = DEFAULT_OPENROUTER_MAX_TOKENS,
        gemini_fallback_models: list[str] | None = None,
        gemini_time_budget_s: float = DEFAULT_GEMINI_TIME_BUDGET_S,
    ) -> None:
        # Hard cap on output (reasoning included) per OpenRouter call — protects paid budgets.
        self.openrouter_max_tokens = openrouter_max_tokens
        # Filled by validate_openrouter_models() from the catalog.
        self.openrouter_vision_models: list[str] = []
        self.gemini_model = gemini_model or DEFAULT_GEMINI_MODEL
        chain: list[str] = []
        for name in [self.gemini_model, *(gemini_fallback_models or DEFAULT_GEMINI_FALLBACK_MODELS)]:
            name = name.strip().removeprefix("models/")
            if name and name not in chain:
                chain.append(name)
        self.gemini_models = chain
        self.gemini_time_budget_s = gemini_time_budget_s
        self._gemini_cooldown: dict[str, float] = {}
        models = [m.strip() for m in (openrouter_models or DEFAULT_OPENROUTER_MODELS) if m.strip()]
        if len(models) > OPENROUTER_MAX_MODELS:
            logger.warning(
                "OpenRouter supports up to %d fallback models, extra ones are ignored: %s",
                OPENROUTER_MAX_MODELS, models[OPENROUTER_MAX_MODELS:],
            )
            models = models[:OPENROUTER_MAX_MODELS]
        self.openrouter_models = models or DEFAULT_OPENROUTER_MODELS

        self._gemini: genai.Client | None = None
        if gemini_api_key:
            self._gemini = genai.Client(
                api_key=gemini_api_key,
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000),
            )

        self._openrouter: AsyncOpenAI | None = None
        if openrouter_api_key:
            self._openrouter = AsyncOpenAI(
                api_key=openrouter_api_key,
                base_url=OPENROUTER_BASE_URL,
                timeout=REQUEST_TIMEOUT_S,
                max_retries=1,
                default_headers={"X-Title": "Math Notes Telegram Bot"},
            )

        if not self._gemini and not self._openrouter:
            raise RuntimeError("Set at least one of GEMINI_API_KEY / OPENROUTER_API_KEY")

    # ------------------------------------------------------------------ public

    async def generate(
        self,
        mode: Mode,
        prompt: str | None,
        media: list[MediaPart] | None = None,
        force_openrouter: bool = False,
    ) -> LLMResult:
        """Generate a LaTeX body for the given mode with automatic fallback."""
        media = media or []
        prompt = (prompt or "").strip() or DEFAULT_USER_PROMPT[mode]
        openrouter_can_take_media = not media or self.openrouter_accepts(media)

        if force_openrouter:
            if not openrouter_can_take_media:
                raise LLMError(
                    "Через /or можно отправить текст или фото (если модель OpenRouter понимает "
                    "изображения). Аудио и PDF обрабатывает только Gemini: используйте /solve или /note."
                )
            return await self._openrouter_generate(mode, prompt, media)

        if self._gemini is not None:
            result = await self._gemini_chain(mode, prompt, media)
            if result is not None:
                return result
            if not openrouter_can_take_media:
                raise LLMError(
                    "Все модели Gemini сейчас недоступны, а резервный провайдер не принимает такие "
                    "вложения. Попробуйте через пару минут или пришлите задачу текстом."
                )
        elif not openrouter_can_take_media:
            raise LLMError("Обработка аудио и PDF требует GEMINI_API_KEY.")

        logger.info("Falling back to OpenRouter (mode=%s, images=%d)", mode.value, len(media))
        return await self._openrouter_generate(mode, prompt, media)

    def openrouter_accepts(self, media: list[MediaPart]) -> bool:
        """OpenRouter fallback handles images only, and only with vision-capable models."""
        return (
            self._openrouter is not None
            and bool(self.openrouter_vision_models)
            and all(m.mime_type.startswith("image/") for m in media)
        )

    async def fix_latex(self, latex_body: str, error_log: str, prefer_openrouter: bool = False) -> str:
        prompt = (
            "Лог ошибки pdflatex (хвост):\n"
            f"{error_log}\n\n"
            "Фрагмент LaTeX с ошибкой:\n"
            f"{latex_body}"
        )
        result = await self.generate(Mode.FIX, prompt, force_openrouter=prefer_openrouter and bool(self._openrouter))
        return result.text

    async def validate_openrouter_models(self) -> None:
        """Drop model IDs that OpenRouter no longer serves.

        A single unknown ID makes OpenRouter reject the whole request with 400,
        and free models are retired regularly, so we check once at startup.
        """
        if self._openrouter is None:
            return
        try:
            page = await self._openrouter.models.list()
            catalog = {m.id: m for m in page.data}
        except Exception as exc:  # noqa: BLE001 — keep configured list if catalog is unreachable
            logger.warning("Could not fetch OpenRouter model catalog: %s", exc)
            return
        available = set(catalog)
        valid = [m for m in self.openrouter_models if m in available]
        invalid = [m for m in self.openrouter_models if m not in available]
        if invalid:
            logger.warning("OpenRouter models not found and skipped: %s", invalid)
        if not valid:
            valid = [OPENROUTER_SAFE_DEFAULT]
            logger.warning("No valid OpenRouter models configured, using %s", OPENROUTER_SAFE_DEFAULT)
        self.openrouter_models = valid
        self.openrouter_vision_models = [m for m in valid if _accepts_images(catalog.get(m))]
        logger.info(
            "OpenRouter fallback chain: %s (image-capable: %s, max_tokens=%d)",
            valid, self.openrouter_vision_models or "none", self.openrouter_max_tokens,
        )

    async def close(self) -> None:
        if self._openrouter is not None:
            await self._openrouter.close()
        if self._gemini is not None:
            aclose = getattr(self._gemini.aio, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    logger.debug("Gemini client close failed", exc_info=True)

    # ------------------------------------------------------------------ gemini

    async def validate_gemini_models(self) -> None:
        """Drop chain entries that this API key cannot see in Google's model catalog."""
        if self._gemini is None:
            return
        try:
            pager = await self._gemini.aio.models.list()
            available: set[str] = set()
            async for model in pager:
                actions = model.supported_actions or []
                if model.name and (not actions or "generateContent" in actions):
                    available.add(model.name.removeprefix("models/"))
        except Exception as exc:  # noqa: BLE001 — keep configured chain if catalog is unreachable
            logger.warning("Could not fetch Gemini model catalog (%s): %s", _describe_error(exc), exc)
            return
        valid = [m for m in self.gemini_models if m in available]
        missing = [m for m in self.gemini_models if m not in available]
        if missing:
            logger.warning("Gemini models not available for this key, skipped: %s", missing)
        if valid:
            self.gemini_models = valid
        else:
            logger.error("None of the configured Gemini models exist; keeping the list as is")
        logger.info("Gemini chain: %s (time budget %.0fs)", self.gemini_models, self.gemini_time_budget_s)

    async def _gemini_chain(self, mode: Mode, prompt: str, media: list[MediaPart]) -> LLMResult | None:
        """Try Gemini models in order; return None if all failed or were cooling down."""
        started = time.monotonic()
        for model in self.gemini_models:
            now = time.monotonic()
            until = self._gemini_cooldown.get(model, 0.0)
            if until > now:
                logger.info("Skipping %s (cooling down for %.0fs more)", model, until - now)
                continue
            remaining = self.gemini_time_budget_s - (now - started)
            if remaining < 20:
                logger.warning("Gemini time budget exhausted after %.0fs", now - started)
                break
            try:
                return await self._gemini_generate(mode, prompt, media, model, timeout=min(REQUEST_TIMEOUT_S, remaining))
            except Exception as exc:  # noqa: BLE001 — any failure moves on to the next model
                cooldown = _gemini_cooldown_for(exc)
                if cooldown:
                    self._gemini_cooldown[model] = time.monotonic() + cooldown
                logger.warning(
                    "Gemini %s failed (%s)%s: %s", model, _describe_error(exc),
                    f", cooling down {cooldown}s" if cooldown else "", str(exc)[:300],
                )
        return None

    def _thinking_config(self, mode: Mode, model: str) -> types.ThinkingConfig:
        deep = mode is Mode.SOLVE
        model = model.lower()
        if "gemini-2" not in model:
            # Gemini 3+ models use discrete thinking levels.
            return types.ThinkingConfig(thinking_level="HIGH" if deep else "LOW")
        # Gemini 2.5: token budgets (Pro cannot disable thinking entirely).
        if "pro" in model:
            return types.ThinkingConfig(thinking_budget=32768 if deep else 128)
        return types.ThinkingConfig(thinking_budget=24576 if deep else 0)

    async def _gemini_generate(
        self, mode: Mode, prompt: str, media: list[MediaPart], model: str, timeout: float = REQUEST_TIMEOUT_S,
    ) -> LLMResult:
        assert self._gemini is not None
        contents: list[types.Part] = [
            types.Part.from_bytes(data=m.data, mime_type=m.mime_type) for m in media
        ]
        contents.append(types.Part.from_text(text=prompt))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPTS[mode],
            thinking_config=self._thinking_config(mode, model),
            temperature=0.2 if mode is not Mode.NOTE else 0.4,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = await asyncio.wait_for(
            self._gemini.aio.models.generate_content(model=model, contents=contents, config=config),
            timeout=timeout,
        )
        text = (response.text or "").strip()
        if not text:
            reason = None
            if response.candidates:
                reason = response.candidates[0].finish_reason
            raise RuntimeError(f"Gemini returned an empty response (finish_reason={reason})")

        logger.info(
            "Gemini ok: model=%s mode=%s usage=%s",
            model, mode.value, getattr(response, "usage_metadata", None),
        )
        return LLMResult(text=text, provider="Gemini", model=model)

    # -------------------------------------------------------------- openrouter

    async def _openrouter_generate(
        self, mode: Mode, prompt: str, media: list[MediaPart] | None = None,
    ) -> LLMResult:
        if self._openrouter is None:
            raise LLMError("Резервный провайдер OpenRouter не настроен (нет OPENROUTER_API_KEY).")

        media = media or []
        # Image requests go only to models that accept images.
        models = self.openrouter_vision_models if media else self.openrouter_models
        if not models:
            raise LLMError("Ни одна модель OpenRouter в списке не принимает изображения.")

        user_content: str | list[dict] = prompt
        if media:
            user_content = [{"type": "text", "text": prompt}] + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{m.mime_type};base64,{base64.b64encode(m.data).decode()}"},
                }
                for m in media
            ]

        extra_body: dict = {
            "models": models,
            "reasoning": {"effort": "high" if mode is Mode.SOLVE else "low"},
        }
        try:
            response = await self._openrouter.chat.completions.create(
                model=models[0],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[mode]},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=self.openrouter_max_tokens,
                extra_body=extra_body,
            )
        except (OpenAIError, asyncio.TimeoutError, OSError) as exc:
            logger.error("OpenRouter failed (%s): %s", _describe_error(exc), exc)
            raise LLMError("Все провайдеры ИИ сейчас недоступны. Попробуйте через минуту.") from exc

        if not response.choices:
            raise LLMError("OpenRouter вернул пустой ответ. Попробуйте ещё раз.")
        if response.choices[0].finish_reason == "length":
            logger.warning("OpenRouter answer hit max_tokens=%d and may be truncated", self.openrouter_max_tokens)
        text = _strip_reasoning(response.choices[0].message.content or "")
        if not text:
            raise LLMError("OpenRouter вернул пустой ответ. Попробуйте ещё раз.")

        used_model = response.model or self.openrouter_models[0]
        logger.info("OpenRouter ok: model=%s mode=%s usage=%s", used_model, mode.value, response.usage)
        return LLMResult(text=text, provider="OpenRouter", model=used_model)


def _accepts_images(model: object | None) -> bool:
    arch = getattr(model, "architecture", None) if model is not None else None
    if not isinstance(arch, dict):
        return False
    return "image" in (arch.get("input_modalities") or [])


def _gemini_cooldown_for(exc: BaseException) -> int:
    """How long to skip a Gemini model after this error (0 = don't skip)."""
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429:
            return COOLDOWN_RATE_LIMIT_S
        if exc.code == 404:
            return COOLDOWN_NOT_FOUND_S
        if exc.code in (408, 500, 502, 503, 504):
            return COOLDOWN_OVERLOADED_S
        return 0  # 400/403 etc.: request- or key-specific, don't punish the model
    if isinstance(exc, (asyncio.TimeoutError, OSError)):
        return COOLDOWN_OVERLOADED_S
    name = type(exc).__name__
    if any(k in name for k in ("Connect", "Timeout", "Network", "Transport")):
        return COOLDOWN_OVERLOADED_S
    return 0


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, genai_errors.APIError):
        kind = "retryable" if exc.code in LLM_HTTP_STATUS_FALLBACK else "api"
        return f"{kind} HTTP {exc.code}"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    name = type(exc).__name__
    if isinstance(exc, OSError) or any(k in name for k in ("Connect", "Timeout", "Network", "Transport")):
        return f"network {name}"
    return name
