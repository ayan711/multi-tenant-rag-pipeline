# ─────────────────────────────────────────────────────────────────────────────
# config.py — Application Configuration & Environment Variable Loader
# ─────────────────────────────────────────────────────────────────────────────
#
# All secrets and connection strings live in a `.env` file on disk and are
# NEVER hardcoded in source code.  This module reads them once at startup,
# validates their types, and exposes a single `settings` object that every
# other module imports.
#
# Why pydantic-settings over plain os.environ?
#   If a required variable is missing or the wrong type, the app crashes
#   immediately at startup with a clear ValidationError — not deep inside
#   a request handler with a confusing AttributeError.
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Tell pydantic-settings where to find the .env file.
    # The path is relative to the current working directory — which is the
    # project root when you run `uvicorn app.main:app` from enterprise-rag/.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Treat OPENAI_API_KEY and openai_api_key as the same variable.
        case_sensitive=False,
    )

    # ── Secrets ──────────────────────────────────────────────────────────────
    #
    # SecretStr wraps the key so that str(settings.gemini_api_key) prints
    # '**********' instead of the real value.  This prevents the key from
    # leaking into logs, tracebacks, or debug output.
    #
    # When you actually need the raw string (e.g. to pass to the OpenAI-SDK
    # client pointed at Gemini), call:  settings.gemini_api_key.get_secret_value()
    #
    # No default value → pydantic raises ValidationError at startup if this
    # variable is missing from the environment.
    gemini_api_key: SecretStr

    # ── Infrastructure connection coordinates ─────────────────────────────────
    #
    # Defaults match the ports defined in docker-compose.yml.
    # Change these only if you edited the left-hand side of the port mappings.

    # Full Redis URL consumed directly by Celery's broker_url and result_backend.
    # Format:  redis://HOST:PORT/DB_INDEX
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB is accessed via its HTTP API (chromadb.HttpClient), so we need
    # host and port separately rather than a single URL string.
    chroma_host: str = "localhost"

    # Field(...) lets us attach extra validation rules.  ge=1 le=65535 ensures
    # the port is a valid TCP port number.
    chroma_port: int = Field(default=8000, ge=1, le=65535)

    # ── Application constants ─────────────────────────────────────────────────
    #
    # These rarely change but are centralised here so they can be overridden
    # via environment variables without touching source code — useful when
    # running experiments (e.g. swapping embedding models).

    # The single ChromaDB collection that holds all tenants' vectors.
    # Multi-tenancy is enforced via metadata filters, not separate collections.
    chroma_collection: str = "enterprise_knowledge_base"

    # Local sentence-transformers model.  Pulls ~90 MB on first use, then
    # cached in ~/.cache/huggingface/hub/.
    embedding_model: str = "all-MiniLM-L6-v2"

    # Gemini model used for answer synthesis at query time, called through
    # Google's OpenAI-compatible endpoint (see gemini_base_url below) so the
    # rest of the app can keep using the `openai` SDK's client/streaming API.
    gemini_model: str = "gemini-3.5-flash"

    # Google's OpenAI-compatibility base URL — swapping providers again later
    # only means changing this + gemini_api_key, not the client code.
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    lru_cache ensures the .env file is read from disk exactly once, no matter
    how many modules call get_settings().

    This function is also used as a FastAPI dependency (Task 5.5):
        Depends(get_settings)
    which makes it easy to override settings in tests by overriding the
    dependency — without monkey-patching global state.
    """
    return Settings()


# Module-level singleton for direct imports used by most modules:
#
#   from app.config import settings
#   client = OpenAI(api_key=settings.gemini_api_key.get_secret_value(), base_url=settings.gemini_base_url)
#
settings = get_settings()
