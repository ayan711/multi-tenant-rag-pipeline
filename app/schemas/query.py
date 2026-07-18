from pydantic import BaseModel, field_validator


class QueryRequest(BaseModel):
    query: str
    tenant_id: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        # A whitespace-only string is still "truthy" to Pydantic's required-field
        # check — without this, embedding it produces a meaningless vector and the
        # failure only surfaces later as an odd/empty answer, not a clear 422.
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_must_not_be_blank(cls, v: str) -> str:
        # Same reasoning as above: a blank tenant_id would pass the where= filter
        # (Task 3.4) as a valid-looking string but match zero vectors, surfacing
        # as a confusing 404 instead of a clear validation error.
        if not v.strip():
            raise ValueError("tenant_id must not be blank")
        return v.strip()


class ContextChunk(BaseModel):
    text: str
    page_number: int
    # Cosine distance from the query vector — surfaced to the client (and used
    # by scripts/eval_retrieval_quality.py) as a rough relevance signal, not
    # just an internal ranking detail (Task 6.5).
    distance: float
