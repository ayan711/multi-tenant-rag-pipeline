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
#
# Error handling: a provider that raises during construction (e.g. ChromaDB
# unreachable) fails BEFORE the route body runs — FastAPI resolves Depends()
# ahead of the endpoint function. Left uncaught, that surfaces as a raw 500
# from main.py's catch-all handler, indistinguishable from a real app bug.
# Providers that make a network call at construction time catch that failure
# and re-raise as HTTPException(503) instead, so callers get a clean signal
# that the dependency is down rather than that the request was malformed.
# lru_cache never caches a raised exception, so the next request retries
# construction fresh once the dependency recovers.
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache

import redis.asyncio as aioredis
from celery import Celery
from fastapi import HTTPException, status
from openai import AsyncOpenAI

from app.celery_app import celery_app
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
    try:
        return VectorStoreManager()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is unavailable",
        ) from exc


@lru_cache(maxsize=1)
def get_llm_client() -> AsyncOpenAI:
    # AsyncOpenAI is an httpx-based client; one instance is reused across requests.
    # base_url points it at Gemini's OpenAI-compatible endpoint instead of OpenAI's —
    # the rest of the app (synthesis.py) is unaware of the swap.
    return AsyncOpenAI(
        api_key=settings.gemini_api_key.get_secret_value(),
        base_url=settings.gemini_base_url,
    )


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    # from_url parses redis://host:port/db — no connection is made until first command.
    return aioredis.Redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache(maxsize=1)
def get_celery_app() -> Celery:
    # celery_app is already a module-level singleton (app/celery_app.py);
    # this just gives routes a swappable seam for dependency_overrides in tests.
    return celery_app
