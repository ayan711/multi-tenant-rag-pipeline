# ─────────────────────────────────────────────────────────────────────────────
# celery_app.py — Bare Celery Application Instance
# ─────────────────────────────────────────────────────────────────────────────
#
# Split out from worker.py so the FastAPI process can import `celery_app` for
# task-status lookups (Task 5.9) without also importing worker.py's heavy
# module-level singletons (LocalEmbedder, VectorStoreManager) — those only
# need to exist inside the Celery worker process, not the API process.
# ─────────────────────────────────────────────────────────────────────────────

from celery import Celery

from app.config import settings

# broker_url      — Redis queue where FastAPI pushes task messages.
# result_backend  — same Redis instance; the worker writes status + return
#                    value here so callers can check progress via
#                    AsyncResult(task_id).state / .result.
celery_app = Celery(
    "enterprise_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# task_serializer / result_serializer / accept_content:
#   JSON is explicit, human-readable, and safe against pickle deserialization
#   attacks that can execute arbitrary code. Celery's default is pickle;
#   we override it here.
#
# task_track_started:
#   By default a task's state jumps from PENDING → SUCCESS (or FAILURE).
#   Setting this to True adds a STARTED state so the status-polling endpoint
#   (Task 5.9) can distinguish "queued" from "actively running".
#
# timezone / enable_utc:
#   Store all timestamps in UTC so logs are unambiguous regardless of where
#   the worker process is running.
#
# task_store_eager_result:
#   Only takes effect when task_always_eager is also on (real dispatch through
#   a live worker is unaffected). Needed for Task 5.9's tests: a Task's
#   store_eager_result flag is copied from this config the first time the task
#   is bound to the app (module import), then frozen — flipping this config
#   later in a test fixture is too late to matter. Setting it permanently here
#   means an eager .delay() actually writes to Redis, so a separate
#   AsyncResult(task_id) lookup afterward (what the endpoint does) can find it.
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_store_eager_result=True,
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # drop completed task results from Redis after 1 hour
)
