"""Sidecar metadata persisted for LLM responses."""

import hashlib
import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from openhands.sdk.conversation.persistence_const import LLM_RESPONSES_DIR
from openhands.sdk.llm.llm_response import LLMLogprobs


_SAFE_RESPONSE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_SAFE_RESPONSE_ID_LENGTH = 80


class LLMResponseMetadata(BaseModel):
    """Persisted metadata for an LLM response, stored outside the event log."""

    schema_version: int = Field(default=1)
    llm_response_id: str
    logprobs: LLMLogprobs | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")


def llm_response_metadata_path(llm_response_id: str) -> str:
    """Return the sidecar path for an LLM response id."""

    safe_id = _SAFE_RESPONSE_ID_RE.sub("_", llm_response_id).strip("._")
    safe_id = safe_id[:_MAX_SAFE_RESPONSE_ID_LENGTH] or "response"
    digest = hashlib.sha256(llm_response_id.encode("utf-8")).hexdigest()[:12]
    return f"{LLM_RESPONSES_DIR}/{safe_id}-{digest}.json"
