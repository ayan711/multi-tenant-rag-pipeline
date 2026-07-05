# ─────────────────────────────────────────────────────────────────────────────
# dependencies.py — FastAPI Dependency Providers
# ─────────────────────────────────────────────────────────────────────────────
#
# Singletons that are expensive to construct (model weights, HTTP connections)
# should be created once per process and reused across requests.
#
# Pattern: a provider function decorated with @lru_cache(maxsize=1).
#   - First call: constructs the object (slow path).
#   - Every subsequent call: returns the cached instance instantly.
#   - FastAPI Depends() calls the function for every request; lru_cache absorbs
#     all calls after the first into a no-op cache lookup.
#
# Testing: FastAPI's app.dependency_overrides replaces the provider function
# entirely, so tests never touch the cache or construct real objects.
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache

import redis.asyncio as aioredis
from openai import AsyncOpenAI

from app.config import settings
from app.embedder import LocalEmbedder
from app.vector_db import VectorStoreManager


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    # Model weights (~80 MB) are loaded from disk on first call and cached.
    return LocalEmbedder()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreManager:
    # Opens an HTTP connection to the ChromaDB server on first call.
    return VectorStoreManager()


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    # AsyncOpenAI is an httpx-based client; one instance is reused across requests.
    return AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    # from_url parses redis://host:port/db — no connection is made until first command.
    return aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
