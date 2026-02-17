import asyncio
import contextlib
import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


Callback = Callable[..., Optional[Awaitable[None]]]


@dataclass
class SpeechChunk:
    session_id: int
    text: str


class SpeakerManager:
    """Session-aware async speaker with barge-in cancellation."""

    def __init__(self, backend, on_enqueued: Optional[Callback] = None, on_spoken: Optional[Callback] = None):
        self.backend = backend
        self.on_enqueued = on_enqueued
        self.on_spoken = on_spoken

        self._session_id = 0
        self._queue: asyncio.Queue[SpeechChunk] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def session_id(self) -> int:
        return self._session_id

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

    async def new_session(self) -> int:
        async with self._lock:
            self._session_id += 1
            sid = self._session_id
            self._clear_queue()
            await self.backend.cancel(sid)
            return sid

    async def enqueue(self, text: str, session_id: int) -> bool:
        clean = text.strip()
        if not clean or session_id != self._session_id:
            return False

        chunk = SpeechChunk(session_id=session_id, text=clean)
        await self._queue.put(chunk)
        await self._emit(self.on_enqueued, chunk)
        return True

    async def wait_idle(self, session_id: int) -> None:
        while session_id == self._session_id:
            if self._queue.empty() and await self.backend.is_idle(session_id):
                return
            await asyncio.sleep(0.03)

    def _clear_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _worker(self) -> None:
        while True:
            chunk = await self._queue.get()
            if chunk.session_id != self._session_id:
                continue
            try:
                await self.backend.speak(chunk.text, chunk.session_id)
                await self._emit(self.on_spoken, chunk)
            except Exception:
                # Keep speaker loop alive for stream runtime.
                continue

    async def _emit(self, cb: Optional[Callback], *args) -> None:
        if not cb:
            return
        result = cb(*args)
        if inspect.isawaitable(result):
            await result
