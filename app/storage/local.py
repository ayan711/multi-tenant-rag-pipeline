import tempfile
from functools import lru_cache
from pathlib import Path

from app.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Stores uploaded files in the OS temp directory.

    Files are named <sha256>.pdf so identical uploads map to the same
    path — no duplicate writes. The Celery worker receives the absolute
    path as its locator and opens the file directly from disk.

    To swap in S3: implement StorageBackend, return an s3://bucket/key
    locator from save(), and register a new get_s3_storage() factory.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        self._base = Path(base_dir or tempfile.gettempdir())

    def save(self, file_hash: str, data: bytes) -> str:
        path = self._base / f"{file_hash}.pdf"
        if not path.exists():
            path.write_bytes(data)
        return str(path)

    def exists(self, file_hash: str) -> bool:
        return (self._base / f"{file_hash}.pdf").exists()


@lru_cache
def get_local_storage() -> LocalStorage:
    """Singleton factory — FastAPI Depends() calls this once per process."""
    return LocalStorage()
