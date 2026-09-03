from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Contract for file storage backends.

    The upload route calls save() and passes the returned locator to the
    Celery worker. Local storage returns an absolute path; an S3 backend
    would return an s3://bucket/key URI. The worker only needs to know
    how to open its own backend's locator format.
    """

    @abstractmethod
    def save(self, file_hash: str, data: bytes) -> str:
        """Persist data keyed by file_hash and return a locator string.

        Implementations must be idempotent: if the same hash has already
        been saved, skip the write and return the existing locator.
        """
        ...

    @abstractmethod
    def exists(self, file_hash: str) -> bool:
        """Return True if a file with this hash has already been stored."""
        ...
