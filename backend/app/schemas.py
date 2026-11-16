import re

from pydantic import BaseModel, Field

NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    namespace: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    session_id: str | None = Field(default=None, max_length=128)
