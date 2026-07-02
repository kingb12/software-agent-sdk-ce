"""LLMResponse type for LLM completion responses.

This module provides the LLMResponse type that wraps LLM completion responses
with OpenHands-native types, eliminating the need for consumers to work directly
with LiteLLM types.
"""

import warnings
from typing import Any, ClassVar

from litellm import ResponsesAPIResponse
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, ConfigDict, Field

from openhands.sdk.llm.message import Message
from openhands.sdk.llm.utils.metrics import MetricsSnapshot


# Suppress Pydantic serializer warnings from litellm
# These warnings occur when Pydantic serializes litellm's ModelResponse objects
# that have mismatched field counts, which is expected behavior in litellm
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


__all__ = ["LLMResponse", "LLMLogprobs", "LLMTokenLogprob", "LLMTopLogprob"]


def _get_logprob_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class LLMTopLogprob(BaseModel):
    """Alternative token and log probability at a generated token position."""

    token: str
    logprob: float

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    @classmethod
    def from_raw(cls, raw: Any) -> "LLMTopLogprob | None":
        token = _get_logprob_value(raw, "token")
        logprob = _get_logprob_value(raw, "logprob")
        if token is None or logprob is None:
            return None
        return cls(
            token=str(token),
            logprob=float(logprob),
        )


class LLMTokenLogprob(BaseModel):
    """Generated token log probability plus optional top alternatives."""

    token: str
    logprob: float
    top_logprobs: list[LLMTopLogprob] = Field(default_factory=list)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    @classmethod
    def from_raw(cls, raw: Any) -> "LLMTokenLogprob | None":
        token = _get_logprob_value(raw, "token")
        logprob = _get_logprob_value(raw, "logprob")
        if token is None or logprob is None:
            return None

        top_logprobs = [
            top_logprob
            for item in (_get_logprob_value(raw, "top_logprobs") or [])
            if (top_logprob := LLMTopLogprob.from_raw(item)) is not None
        ]
        return cls(
            token=str(token),
            logprob=float(logprob),
            top_logprobs=top_logprobs,
        )


class LLMLogprobs(BaseModel):
    """Output-token log probabilities returned by an LLM response."""

    content: list[LLMTokenLogprob] = Field(default_factory=list)
    refusal: list[LLMTokenLogprob] = Field(default_factory=list)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    @classmethod
    def from_chat_logprobs(cls, raw: Any) -> "LLMLogprobs | None":
        if raw is None:
            return None
        logprobs = cls(
            content=cls._tokens_from_raw(_get_logprob_value(raw, "content")),
            refusal=cls._tokens_from_raw(_get_logprob_value(raw, "refusal")),
        )
        return logprobs if logprobs.content or logprobs.refusal else None

    @classmethod
    def from_responses_output(cls, output: Any) -> "LLMLogprobs | None":
        content: list[LLMTokenLogprob] = []
        for item in output or []:
            if _get_logprob_value(item, "type") != "message":
                continue
            for part in _get_logprob_value(item, "content") or []:
                if _get_logprob_value(part, "type") != "output_text":
                    continue
                content.extend(
                    cls._tokens_from_raw(_get_logprob_value(part, "logprobs"))
                )

        return cls(content=content) if content else None

    @staticmethod
    def _tokens_from_raw(raw_tokens: Any) -> list[LLMTokenLogprob]:
        return [
            token_logprob
            for item in (raw_tokens or [])
            if (token_logprob := LLMTokenLogprob.from_raw(item)) is not None
        ]


class LLMResponse(BaseModel):
    """Result of an LLM completion request.

    This type provides a clean interface for LLM completion results, exposing
    only OpenHands-native types to consumers while preserving access to the
    raw LiteLLM response for internal use.

    Attributes:
        message: The completion message converted to OpenHands Message type
        metrics: Snapshot of metrics from the completion request
        raw_response: The original LiteLLM response (ModelResponse or
            ResponsesAPIResponse) for internal use
    """

    message: Message
    metrics: MetricsSnapshot
    raw_response: ModelResponse | ResponsesAPIResponse
    logprobs: LLMLogprobs | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @property
    def id(self) -> str:
        """Get the response ID from the underlying LLM response.

        This property provides a clean interface to access the response ID,
        supporting both completion mode (ModelResponse) and response API modes
        (ResponsesAPIResponse).

        Returns:
            The response ID from the LLM response
        """
        return self.raw_response.id
