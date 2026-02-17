from typing import List


class TokenChunker:
    """Convert token deltas into speakable chunks with low initial latency."""

    def __init__(
        self,
        early_flush_chars: int = 24,
        sentence_flush_chars: int = 45,
        hard_cap_chars: int = 230,
    ):
        self.early_flush_chars = early_flush_chars
        self.sentence_flush_chars = sentence_flush_chars
        self.hard_cap_chars = hard_cap_chars
        self._buffer = ""
        self._did_first_flush = False

    def reset(self) -> None:
        self._buffer = ""
        self._did_first_flush = False

    def push(self, token: str) -> List[str]:
        if not token:
            return []

        emitted: List[str] = []
        self._buffer += token

        if len(self._buffer) >= self.hard_cap_chars:
            emitted.append(self._flush())
            return emitted

        if not self._did_first_flush and len(self._buffer) >= self.early_flush_chars:
            if self._buffer.endswith(" ") or self._buffer.endswith(","):
                emitted.append(self._flush())
                return emitted

        ends_sentence = any(self._buffer.rstrip().endswith(p) for p in (".", "!", "?", "…", "\n"))
        if ends_sentence and len(self._buffer.strip()) >= self.sentence_flush_chars:
            emitted.append(self._flush())

        return emitted

    def finish(self) -> List[str]:
        if self._buffer.strip():
            return [self._flush()]
        return []

    def _flush(self) -> str:
        text = self._buffer.strip()
        self._buffer = ""
        self._did_first_flush = True
        return text
