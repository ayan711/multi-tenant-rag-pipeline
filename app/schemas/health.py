from pydantic import BaseModel


class HealthResponse(BaseModel):
    redis: str   # "ok" or "error: <detail>"
    chroma: str  # "ok" or "error: <detail>"
