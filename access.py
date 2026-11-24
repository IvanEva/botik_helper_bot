"""Access control: the bot serves only its owner(s) and members of allowed groups.

Rules (active once ALLOWED_USER_IDS or ALLOWED_CHAT_IDS is set):

* Groups: the bot works only in chats listed in ALLOWED_CHAT_IDS. If someone
  else adds it to another group, it leaves immediately. If an owner adds it,
  it stays and posts the group ID so it can be put into ALLOWED_CHAT_IDS.
* Private chats: allowed for ALLOWED_USER_IDS and for anyone who is currently
  a member of an allowed group (checked via getChatMember, cached).
* ``/id`` always works so the owner can discover user and chat IDs.

With both variables empty the bot is open to everyone (a warning is logged).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramAPIError
from aiogram.types import ChatMemberUpdated, Message

logger = logging.getLogger(__name__)

POSITIVE_TTL_S = 600  # members keep access for 10 min without re-checking
NEGATIVE_TTL_S = 60   # newcomers to the group get access within a minute

_MEMBER_STATUSES = {ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.MEMBER}
_GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
_ID_COMMAND_RE = re.compile(r"^/id(?:@\w+)?(?:\s|$)", re.IGNORECASE)


def parse_ids(raw: str | None) -> set[int]:
    ids: set[int] = set()
    for chunk in re.split(r"[,\s;]+", raw or ""):
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            logger.warning("Ignoring invalid Telegram ID %r", chunk)
    return ids


class AccessControl:
    def __init__(self, allowed_users: set[int], allowed_chats: set[int]) -> None:
        self.allowed_users = set(allowed_users)
        self.allowed_chats = set(allowed_chats)
        self._cache: dict[int, tuple[bool, float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.allowed_users or self.allowed_chats)

    def is_owner(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.allowed_users

    def is_chat_allowed(self, chat_id: int) -> bool:
        return not self.enabled or chat_id in self.allowed_chats

    def migrate_chat(self, old_id: int, new_id: int) -> None:
        """Basic group -> supergroup upgrade changes the chat ID."""
        if old_id in self.allowed_chats and new_id not in self.allowed_chats:
            self.allowed_chats.add(new_id)
            logger.warning(
                "Allowed group %s was upgraded to supergroup %s. Access kept for this run; "
                "put %s into ALLOWED_CHAT_IDS on Render to make it permanent.", old_id, new_id, new_id,
            )

    async def is_user_allowed(self, bot: Bot, user_id: int) -> bool:
        if not self.enabled or user_id in self.allowed_users:
            return True
        now = time.monotonic()
        cached = self._cache.get(user_id)
        if cached and cached[1] > now:
            return cached[0]

        had_error = False
        for chat_id in self.allowed_chats:
            try:
                member = await bot.get_chat_member(chat_id=chat_id, user_id=user_id)
            except TelegramAPIError as exc:
                had_error = True
                logger.warning("getChatMember(%s, %s) failed: %s", chat_id, user_id, exc)
                continue
            if member.status in _MEMBER_STATUSES or (
                member.status == ChatMemberStatus.RESTRICTED and getattr(member, "is_member", False)
            ):
                self._cache[user_id] = (True, now + POSITIVE_TTL_S)
                return True

        if not had_error:  # don't cache a denial caused by a transient API error
            self._cache[user_id] = (False, now + NEGATIVE_TTL_S)
        return False


def _is_command(message: Message) -> bool:
    return (message.text or message.caption or "").startswith("/")


class AccessMiddleware(BaseMiddleware):
    """Outer middleware for messages: drops everything from outsiders."""

    def __init__(self, access: AccessControl) -> None:
        self.access = access

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        access = self.access
        if not access.enabled or _ID_COMMAND_RE.match(event.text or event.caption or ""):
            return await handler(event, data)

        chat = event.chat
        user_id = event.from_user.id if event.from_user else None

        if chat.type == ChatType.PRIVATE:
            if user_id is not None and await access.is_user_allowed(data["bot"], user_id):
                return await handler(event, data)
            logger.info("Denied private access for user %s", user_id)
            await event.answer(
                "⛔ Это приватный бот: он доступен только участникам определённой группы.\n"
                f"Ваш ID: {user_id} — передайте его владельцу, если нужен доступ."
            )
            return None

        if chat.type in _GROUP_TYPES:
            if event.migrate_to_chat_id:
                access.migrate_chat(chat.id, event.migrate_to_chat_id)
            if event.migrate_from_chat_id:
                access.migrate_chat(event.migrate_from_chat_id, chat.id)
            if access.is_chat_allowed(chat.id):
                return await handler(event, data)
            if _is_command(event):
                logger.info("Denied command in chat %s (%s) from user %s", chat.id, chat.title, user_id)
                if access.is_owner(user_id):
                    await event.reply(
                        f"Эта группа ещё не разрешена. Её ID: {chat.id}\n"
                        "Добавьте его в переменную ALLOWED_CHAT_IDS на Render."
                    )
                else:
                    await event.reply("⛔ Бот не обслуживает эту группу.")
            return None

        return None  # channels and anything else


async def handle_bot_added(event: ChatMemberUpdated, bot: Bot, access: AccessControl) -> None:
    """React to the bot being added to a group."""
    if event.chat.type not in _GROUP_TYPES or not access.enabled:
        return
    if event.new_chat_member.status not in (ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR):
        return
    if event.old_chat_member.status in (ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR):
        return  # e.g. promoted to admin, not a fresh join
    if access.is_chat_allowed(event.chat.id):
        logger.info("Added to allowed group %s (%s)", event.chat.id, event.chat.title)
        return

    adder = event.from_user.id if event.from_user else None
    if access.is_owner(adder):
        logger.warning("Owner added bot to new group %s (%s)", event.chat.id, event.chat.title)
        await bot.send_message(
            event.chat.id,
            f"Привет! ID этой группы: {event.chat.id}\n"
            "Чтобы я здесь заработал, добавьте его в ALLOWED_CHAT_IDS на Render.",
        )
        return

    logger.warning("Unauthorized user %s added bot to %s (%s) — leaving", adder, event.chat.id, event.chat.title)
    try:
        await bot.send_message(event.chat.id, "⛔ Это приватный бот, он не работает в сторонних группах.")
        await bot.leave_chat(event.chat.id)
    except TelegramAPIError as exc:
        logger.warning("Failed to leave chat %s: %s", event.chat.id, exc)
