"""Telegram bot entry point: aiogram polling + aiohttp healthcheck server.

Commands: /start, /help, /solve, /note, /or.
Private chats: any text / photo / voice / audio is processed automatically.
Groups: only explicit commands (also as photo/audio caption or as a reply).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BufferedInputFile, Message
from aiogram.utils.chat_action import ChatActionSender
from aiohttp import ClientSession, ClientTimeout, web
from dotenv import load_dotenv

from access import AccessControl, AccessMiddleware, handle_bot_added, parse_ids
from llm_service import (
    DEFAULT_GEMINI_FALLBACK_MODELS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_TIME_BUDGET_S,
    DEFAULT_OPENROUTER_MAX_TOKENS,
    DEFAULT_OPENROUTER_MODELS,
    LLMError,
    LLMResult,
    LLMService,
    MediaPart,
    Mode,
)
from pdf_builder import build_pdf

logger = logging.getLogger("bot")

TELEGRAM_MSG_LIMIT = 4000
MAX_FILE_BYTES = 20 * 1024 * 1024  # Bot API getFile limit
RATE_LIMIT_SECONDS = 15
MAX_CONCURRENT_JOBS = 2
SUPPORTED_DOC_MIME_PREFIXES = ("image/", "audio/")
SUPPORTED_DOC_MIME_EXACT = ("application/pdf",)

HELP_TEXT = (
    "<b>Бот по высшей математике</b> 📐\n\n"
    "Решаю задачи и делаю конспекты, отвечаю готовым PDF (LaTeX).\n\n"
    "<b>Команды</b>\n"
    "/solve &lt;задача&gt; — подробное решение (глубокое рассуждение)\n"
    "/note &lt;тема&gt; — структурированный конспект\n"
    "/or &lt;запрос&gt; — принудительно через OpenRouter (только текст)\n"
    "/id — показать ваш ID и ID группы\n"
    "/help — эта справка\n\n"
    "<b>Вложения</b>\n"
    "• Фото/скрин задачи → решение\n"
    "• Голосовое или аудио лекции → конспект\n"
    "• Можно ответить командой на сообщение с фото/аудио\n\n"
    "В личке просто пришлите текст, фото или голосовое. "
    "В группах я реагирую только на команды.\n"
    f"Лимит: 1 запрос в {RATE_LIMIT_SECONDS} секунд."
)

_SOLVE_HINT_RE = re.compile(
    r"(реши|решить|решение|найд|вычисл|докаж|доказать|проинтегр|продиффер|упрост|"
    r"исследуй|определи|составь уравнение|сколько|чему равн|lim|\\int|\\frac|∫|"
    r"\d\s*[-+*/^=]\s*\d|[a-zA-Z]\s*\^\s*\d|\bdx\b|y'|y''|=)",
    re.IGNORECASE,
)
_NOTE_HINT_RE = re.compile(r"(конспект|объясни|расскажи|что такое|теори|лекци|определение)", re.IGNORECASE)


def detect_mode(text: str | None, has_image: bool = False, has_audio: bool = False) -> Mode:
    """Guess whether the user wants a solution or notes."""
    text = text or ""
    if _NOTE_HINT_RE.search(text):
        return Mode.NOTE
    if has_audio:
        return Mode.NOTE
    if has_image or _SOLVE_HINT_RE.search(text):
        return Mode.SOLVE
    return Mode.NOTE


def split_text(text: str, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    """Split text into Telegram-sized chunks, preferring newline boundaries."""
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Settings:
    telegram_token: str
    gemini_api_key: str | None
    openrouter_api_key: str | None
    gemini_model: str
    openrouter_models: list[str]
    openrouter_max_tokens: int
    gemini_fallback_models: list[str]
    gemini_time_budget_s: float
    port: int
    self_ping_url: str | None
    self_ping_interval: float
    allowed_user_ids: set[int]
    allowed_chat_ids: set[int]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        token = os.getenv("TELEGRAM_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_TOKEN is not set")
        models_raw = os.getenv("OPENROUTER_FALLBACK_MODELS", "").strip()
        models = [m.strip() for m in models_raw.split(",") if m.strip()] or list(DEFAULT_OPENROUTER_MODELS)
        return cls(
            telegram_token=token,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip() or None,
            gemini_model=os.getenv("PRIMARY_GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL,
            openrouter_models=models,
            openrouter_max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", str(DEFAULT_OPENROUTER_MAX_TOKENS))),
            gemini_fallback_models=[
                m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()
            ] or list(DEFAULT_GEMINI_FALLBACK_MODELS),
            gemini_time_budget_s=float(os.getenv("GEMINI_TIME_BUDGET_S", str(DEFAULT_GEMINI_TIME_BUDGET_S))),
            port=int(os.getenv("PORT", "10000")),
            # Render sets RENDER_EXTERNAL_URL automatically for web services.
            self_ping_url=(os.getenv("SELF_PING_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").strip() or None,
            self_ping_interval=float(os.getenv("SELF_PING_INTERVAL", "600")),
            allowed_user_ids=parse_ids(os.getenv("ALLOWED_USER_IDS")),
            allowed_chat_ids=parse_ids(os.getenv("ALLOWED_CHAT_IDS")),
        )


# --------------------------------------------------------------------------- #
# Rate limiting & media
# --------------------------------------------------------------------------- #


@dataclass
class RateLimiter:
    interval: float = RATE_LIMIT_SECONDS
    _last: dict[int, float] = field(default_factory=dict)

    def check(self, user_id: int) -> float:
        """Return 0 if allowed (and record the hit), else seconds to wait."""
        now = time.monotonic()
        last = self._last.get(user_id)
        if last is not None and now - last < self.interval:
            return self.interval - (now - last)
        self._last[user_id] = now
        if len(self._last) > 10_000:  # prune stale entries
            self._last = {uid: ts for uid, ts in self._last.items() if now - ts < self.interval}
        return 0.0


@dataclass(slots=True)
class MediaRef:
    file_id: str
    mime_type: str
    size: int | None
    kind: str  # "image" | "audio" | "pdf"


def extract_media(message: Message | None) -> MediaRef | None:
    if message is None:
        return None
    if message.photo:
        p = message.photo[-1]
        return MediaRef(p.file_id, "image/jpeg", p.file_size, "image")
    if message.voice:
        v = message.voice
        return MediaRef(v.file_id, v.mime_type or "audio/ogg", v.file_size, "audio")
    if message.audio:
        a = message.audio
        return MediaRef(a.file_id, a.mime_type or "audio/mpeg", a.file_size, "audio")
    if message.document and message.document.mime_type:
        d = message.document
        mime = d.mime_type
        if mime.startswith(SUPPORTED_DOC_MIME_PREFIXES) or mime in SUPPORTED_DOC_MIME_EXACT:
            kind = "image" if mime.startswith("image/") else "audio" if mime.startswith("audio/") else "pdf"
            return MediaRef(d.file_id, mime, d.file_size, kind)
    return None


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #


class TaskProcessor:
    def __init__(self, llm: LLMService) -> None:
        self.llm = llm
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        self.rate_limiter = RateLimiter()

    async def handle(
        self,
        message: Message,
        bot: Bot,
        text: str | None,
        mode: Mode | None = None,
        force_openrouter: bool = False,
    ) -> None:
        # Media: from the message itself, else from the replied-to message.
        reply = message.reply_to_message
        media_ref = extract_media(message) or extract_media(reply)
        text = (text or "").strip()
        if not text and reply is not None and not extract_media(message):
            text = (reply.text or reply.caption or "").strip()

        if not text and media_ref is None:
            await message.reply(
                "Пришлите задачу текстом, фото или голосовым. Например:\n"
                "/solve найти предел lim (sin x)/x при x→0"
            )
            return

        if force_openrouter and media_ref is not None:
            images_ok = media_ref.kind == "image" and bool(self.llm.openrouter_vision_models)
            if not images_ok:
                if not text:
                    await message.reply(
                        "/or принимает текст и фото (если модель OpenRouter понимает изображения). "
                        "Для аудио и PDF используйте /solve или /note."
                    )
                    return
                media_ref = None  # ignore unsupported attachment, use text only

        if media_ref is not None and media_ref.size and media_ref.size > MAX_FILE_BYTES:
            await message.reply("Файл больше 20 МБ — Telegram не даёт ботам скачивать такие файлы.")
            return

        if mode is None:
            mode = detect_mode(
                text,
                has_image=media_ref is not None and media_ref.kind in ("image", "pdf"),
                has_audio=media_ref is not None and media_ref.kind == "audio",
            )

        user_id = message.from_user.id if message.from_user else message.chat.id
        wait = self.rate_limiter.check(user_id)
        if wait > 0:
            await message.reply(f"⏱ Не так быстро: следующий запрос через {int(wait) + 1} сек.")
            return

        if self.semaphore.locked():
            await message.reply("⏳ Все слоты заняты, ваш запрос в очереди…")

        async with self.semaphore:
            await self._run_job(message, bot, text, mode, media_ref, force_openrouter)

    async def _run_job(
        self,
        message: Message,
        bot: Bot,
        text: str,
        mode: Mode,
        media_ref: MediaRef | None,
        force_openrouter: bool,
    ) -> None:
        status_text = "🧠 Решаю задачу…" if mode is Mode.SOLVE else "📝 Составляю конспект…"
        status = await message.reply(status_text)
        started = time.monotonic()
        try:
            thread_id = message.message_thread_id if message.is_topic_message else None
            async with ChatActionSender.upload_document(
                bot=bot, chat_id=message.chat.id, message_thread_id=thread_id,
            ):
                media: list[MediaPart] = []
                if media_ref is not None:
                    buffer = await bot.download(media_ref.file_id, destination=BytesIO())
                    media.append(MediaPart(data=buffer.getvalue(), mime_type=media_ref.mime_type))
                    buffer.close()

                result = await self.llm.generate(mode, text, media, force_openrouter=force_openrouter)
                await _safe_edit(status, f"{status_text}\n🛠 Собираю PDF…")

                async def fixer(body: str, log: str) -> str:
                    await _safe_edit(status, f"{status_text}\n🔧 Исправляю LaTeX…")
                    return await self.llm.fix_latex(body, log, prefer_openrouter=result.provider == "OpenRouter")

                pdf_result = await build_pdf(result.text, fixer=fixer)

            elapsed = time.monotonic() - started
            if pdf_result.ok:
                await self._send_pdf(message, mode, result, pdf_result.pdf, elapsed)
            else:
                await self._send_text_fallback(message, pdf_result.latex)
        except LLMError as exc:
            await message.reply(f"⚠️ {exc}")
        except Exception:  # noqa: BLE001 — never let one job crash the bot
            logger.exception("Job failed for chat %s", message.chat.id)
            await message.reply("⚠️ Внутренняя ошибка при обработке запроса. Попробуйте ещё раз.")
        finally:
            try:
                await status.delete()
            except TelegramBadRequest:
                pass

    @staticmethod
    async def _send_pdf(message: Message, mode: Mode, result: LLMResult, pdf: bytes, elapsed: float) -> None:
        prefix = "Solution" if mode is Mode.SOLVE else "Conspect"
        filename = f"{prefix}_{message.message_id}.pdf"
        label = "Решение" if mode is Mode.SOLVE else "Конспект"
        caption = f"✅ {label} готово · {result.provider} ({result.model}) · {elapsed:.0f} с"
        await message.reply_document(BufferedInputFile(pdf, filename=filename), caption=caption[:1024])
        logger.info("Sent %s (%d bytes) to chat %s", filename, len(pdf), message.chat.id)

    @staticmethod
    async def _send_text_fallback(message: Message, latex: str) -> None:
        await message.reply("⚠️ Не удалось скомпилировать PDF. Отправляю решение текстом (LaTeX):")
        for chunk in split_text(latex):
            await message.reply(chunk)


async def _safe_edit(message: Message, text: str) -> None:
    try:
        await message.edit_text(text)
    except TelegramBadRequest:
        pass


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #

router = Router(name="main")


@router.message(CommandStart())
@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.reply(HELP_TEXT, parse_mode=ParseMode.HTML)


@router.message(Command("id"))
async def cmd_id(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else "?"
    text = f"Ваш ID: <code>{user_id}</code>"
    if message.chat.type != ChatType.PRIVATE:
        text += f"\nID этой группы: <code>{message.chat.id}</code>"
    await message.reply(text, parse_mode=ParseMode.HTML)


router.my_chat_member.register(handle_bot_added)


@router.message(Command("solve"))
async def cmd_solve(message: Message, command: CommandObject, bot: Bot, processor: TaskProcessor) -> None:
    await processor.handle(message, bot, command.args, mode=Mode.SOLVE)


@router.message(Command("note"))
async def cmd_note(message: Message, command: CommandObject, bot: Bot, processor: TaskProcessor) -> None:
    await processor.handle(message, bot, command.args, mode=Mode.NOTE)


@router.message(Command("or"))
async def cmd_or(message: Message, command: CommandObject, bot: Bot, processor: TaskProcessor) -> None:
    await processor.handle(message, bot, command.args, force_openrouter=True)


# Private chats only: free-form input without commands.
_private = Router(name="private")
_private.message.filter(F.chat.type == ChatType.PRIVATE)


@_private.message(F.text.startswith("/"))
async def unknown_command(message: Message) -> None:
    await message.reply("Неизвестная команда. Список команд — /help")


@_private.message(F.text | F.photo | F.voice | F.audio | F.document)
async def private_any(message: Message, bot: Bot, processor: TaskProcessor) -> None:
    if message.document and extract_media(message) is None:
        await message.reply("Поддерживаются изображения, аудио и PDF.")
        return
    await processor.handle(message, bot, message.text or message.caption)


router.include_router(_private)


# --------------------------------------------------------------------------- #
# Healthcheck server & entry point
# --------------------------------------------------------------------------- #


async def health(_: web.Request) -> web.Response:
    return web.Response(text="OK")


async def start_health_server(port: int) -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/", health)  # GET (and HEAD, used by UptimeRobot)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host="0.0.0.0", port=port).start()
    logger.info("Healthcheck server listening on :%d", port)
    return runner


async def self_ping_loop(url: str, interval: float) -> None:
    """Keep a Render free instance awake by hitting its own public URL.

    Requests to the public ``*.onrender.com`` address pass through Render's
    proxy and count as inbound traffic. This is a backup for an external
    monitor (UptimeRobot): a sleeping instance cannot wake itself up.
    """
    target = url.rstrip("/") + "/health"
    logger.info("Self-ping enabled: %s every %.0fs", target, interval)
    timeout = ClientTimeout(total=30)
    async with ClientSession(timeout=timeout) as session:
        while True:
            await asyncio.sleep(interval)
            try:
                async with session.get(target) as resp:
                    if resp.status != 200:
                        logger.warning("Self-ping got HTTP %s", resp.status)
                    else:
                        logger.debug("Self-ping OK")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — never let the pinger die
                logger.warning("Self-ping failed: %s", exc)


def start_self_ping(settings: "Settings") -> asyncio.Task | None:
    if not settings.self_ping_url or settings.self_ping_interval <= 0:
        logger.info("Self-ping disabled (no RENDER_EXTERNAL_URL / SELF_PING_URL, or SELF_PING_INTERVAL=0)")
        return None
    # Render sleeps after 15 min idle; keep the interval well below that.
    interval = min(max(settings.self_ping_interval, 60.0), 840.0)
    return asyncio.create_task(self_ping_loop(settings.self_ping_url, interval), name="self-ping")


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = Settings.from_env()
    llm = LLMService(
        gemini_api_key=settings.gemini_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        gemini_model=settings.gemini_model,
        openrouter_models=settings.openrouter_models,
        openrouter_max_tokens=settings.openrouter_max_tokens,
        gemini_fallback_models=settings.gemini_fallback_models,
        gemini_time_budget_s=settings.gemini_time_budget_s,
    )
    logger.info(
        "Providers: Gemini=%s (%s), OpenRouter=%s %s",
        bool(settings.gemini_api_key), settings.gemini_model,
        bool(settings.openrouter_api_key), llm.openrouter_models,
    )

    access = AccessControl(settings.allowed_user_ids, settings.allowed_chat_ids)
    if access.enabled:
        logger.info(
            "Access restricted: owners=%s, groups=%s",
            sorted(access.allowed_users), sorted(access.allowed_chats),
        )
    else:
        logger.warning("Access is OPEN to everyone: set ALLOWED_USER_IDS / ALLOWED_CHAT_IDS to restrict")

    bot = Bot(token=settings.telegram_token)
    dp = Dispatcher(processor=TaskProcessor(llm), access=access)
    dp.message.outer_middleware(AccessMiddleware(access))
    dp.include_router(router)

    # Start healthcheck first so Render sees an open port quickly.
    runner = await start_health_server(settings.port)
    ping_task = start_self_ping(settings)
    try:
        await llm.validate_gemini_models()
        await llm.validate_openrouter_models()
        await bot.delete_webhook(drop_pending_updates=False)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        if ping_task is not None:
            ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ping_task
        await runner.cleanup()
        await llm.close()
        await bot.session.close()
        logger.info("Bot stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
