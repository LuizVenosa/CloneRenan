import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Set


@dataclass
class ChatMessage:
    user: str
    text: str
    timestamp: float
    is_mod: bool = False
    is_streamer: bool = False


class TwitchAdapter:
    """Async Twitch chat adapter using TwitchIO."""

    def __init__(self, token: str, client_id: str, nick: str, prefix: str = "!ask"):
        self.token = token
        self.client_id = client_id
        self.nick = nick
        self.prefix = prefix

    async def start(
        self,
        channel: str,
        out_queue: asyncio.Queue,
        user_cooldown_seconds: float = 15.0,
        priority_users: Optional[Set[str]] = None,
    ) -> None:
        try:
            from twitchio.ext import commands
        except Exception as exc:
            raise RuntimeError("TwitchIO is not installed. Install with: pip install twitchio") from exc

        priority_users = {u.lower() for u in (priority_users or set())}
        cooldowns: dict[str, float] = {}
        adapter = self

        class _Bot(commands.Bot):
            def __init__(self):
                super().__init__(
                    token=adapter.token,
                    client_id=adapter.client_id,
                    nick=adapter.nick,
                    prefix="!",
                    initial_channels=[channel],
                )

            async def event_message(self, message):
                if message.echo or not message.content:
                    return

                raw_user = message.author.name if message.author else "unknown"
                user = raw_user.lower()
                text = message.content.strip()
                is_mod = bool(getattr(message.author, "is_mod", False))
                is_streamer = bool(getattr(message.author, "is_broadcaster", False))
                is_priority = is_mod or is_streamer or user in priority_users

                if not text.startswith(adapter.prefix):
                    return

                ask_text = text[len(adapter.prefix) :].strip()
                if not ask_text:
                    return

                now = time.time()
                if not is_priority:
                    last = cooldowns.get(user, 0)
                    if now - last < user_cooldown_seconds:
                        return
                    cooldowns[user] = now

                await out_queue.put(
                    ChatMessage(
                        user=raw_user,
                        text=ask_text,
                        timestamp=now,
                        is_mod=is_mod,
                        is_streamer=is_streamer,
                    )
                )

        bot = _Bot()
        await bot.start()
