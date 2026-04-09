import gzip
from app.core.config import settings


class ChatServices:
    @staticmethod
    def _compress(text: str) -> bytes:
        """UTF-8 encode then gzip-compress."""
        return gzip.compress(text.encode("utf-8"), compresslevel=settings.GZIP_LEVEL)

    @staticmethod
    def _decompress(data: bytes) -> str:
        """Decompress gzip bytes and decode to str."""
        return gzip.decompress(data).decode("utf-8")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Very rough token estimate (≈4 chars per token).
        Replace with tiktoken or your tokeniser if precision matters."""
        return max(1, len(text) // 4)
