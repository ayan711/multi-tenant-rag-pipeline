from pydantic import BaseModel, field_validator


class QueryRequest(BaseModel):
    query: str
    tenant_id: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("tenant_id must not be blank")
        return v.strip()


class ContextChunk(BaseModel):
    text: str
    page_number: int
    distance: float
