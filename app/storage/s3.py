# ── S3 Storage Backend ────────────────────────────────────────────────────────
#
# Drop-in replacement for LocalStorage. Stores uploaded PDFs in an S3 bucket
# keyed by SHA-256 hash. The Celery worker receives an s3://bucket/key locator
# and downloads the file before processing.
#
# ── To enable (4 steps) ───────────────────────────────────────────────────────
#
# 1. Add boto3 to requirements.txt:
#       boto3==1.35.*
#
# 2. Add these three variables to config.py → Settings:
#       s3_bucket: str = "enterprise-rag-uploads"
#       s3_region: str = "us-east-1"
#       aws_access_key_id: SecretStr      # or use an IAM role — no key needed
#       aws_secret_access_key: SecretStr  # same
#    Add them to .env (or your IAM instance profile covers it automatically).
#
# 3. In app/main.py swap the Depends factory on the upload route:
#       from app.storage.s3 import get_s3_storage
#       # in upload.py: storage: StorageBackend = Depends(get_s3_storage)
#
# 4. Update the Celery worker (worker.py) to download the file when the
#    locator starts with "s3://":
#       if path.startswith("s3://"):
#           _, _, rest = path[5:].partition("/")   # strip "s3://"
#           bucket, _, key = rest.partition("/")
#           s3 = boto3.client("s3")
#           tmp = Path(tempfile.gettempdir()) / key
#           s3.download_file(bucket, key, str(tmp))
#           path = str(tmp)
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from functools import lru_cache

# boto3 is intentionally not in requirements.txt yet.
# Uncomment the import below when Step 1 is done.
# import boto3

from app.storage.base import StorageBackend


class S3Storage(StorageBackend):
    """Stores uploaded PDFs in S3, keyed by SHA-256 hash.

    save() is idempotent: head_object checks existence before uploading,
    so the same file is never uploaded twice.

    Returns locator format:  s3://<bucket>/<file_hash>.pdf
    """

    def __init__(self, bucket: str, region: str) -> None:
        self._bucket = bucket
        # self._client = boto3.client("s3", region_name=region)  # Step 1

    def _key(self, file_hash: str) -> str:
        return f"{file_hash}.pdf"

    def save(self, file_hash: str, data: bytes) -> str:
        key = self._key(file_hash)
        if not self.exists(file_hash):
            # self._client.put_object(Bucket=self._bucket, Key=key, Body=data)  # Step 1
            pass
        return f"s3://{self._bucket}/{key}"

    def exists(self, file_hash: str) -> bool:
        # try:
        #     self._client.head_object(Bucket=self._bucket, Key=self._key(file_hash))
        #     return True
        # except self._client.exceptions.ClientError:
        #     return False
        return False  # placeholder until boto3 is wired in


@lru_cache
def get_s3_storage() -> S3Storage:
    """Singleton factory for FastAPI Depends().

    Reads bucket + region from settings — add s3_bucket / s3_region
    to config.py → Settings before using (Step 2).
    """
    # from app.config import settings
    # return S3Storage(
    #     bucket=settings.s3_bucket,
    #     region=settings.s3_region,
    # )
    raise NotImplementedError("Complete Steps 1–2 in app/storage/s3.py before using S3Storage")
