from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from openhands.sdk.agent import Agent
from openhands.sdk.conversation import Conversation
from openhands.sdk.conversation.llm_response_metadata import llm_response_metadata_path
from openhands.sdk.conversation.persistence_const import LLM_RESPONSES_DIR
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.llm import LLMResponse, Message, TextContent
from openhands.sdk.llm.llm_response import LLMLogprobs, LLMTokenLogprob
from openhands.sdk.llm.streaming import AnyTokenCallbackType, TokenCallbackType
from openhands.sdk.testing import TestLLM


if TYPE_CHECKING:
    from openhands.sdk.llm.llm import LLMCallContext
    from openhands.sdk.tool.tool import ToolDefinition


def _message(text: str = "done") -> Message:
    return Message(role="assistant", content=[TextContent(text=text)])


def _logprobs(token: str = "done") -> LLMLogprobs:
    return LLMLogprobs(
        content=[LLMTokenLogprob(token=token, logprob=-0.25)],
    )


class LogprobTestLLM(TestLLM):
    def completion(
        self,
        messages: list[Message],
        tools: Sequence["ToolDefinition"] | None = None,
        add_security_risk_prediction: bool = False,
        on_token: TokenCallbackType | None = None,
        call_context: "LLMCallContext | None" = None,
        **kwargs: Any,
    ) -> LLMResponse:
        response = super().completion(
            messages=messages,
            tools=tools,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token,
            call_context=call_context,
            **kwargs,
        )
        return response.model_copy(update={"logprobs": _logprobs("agent-token")})

    async def acompletion(
        self,
        messages: list[Message],
        tools: Sequence["ToolDefinition"] | None = None,
        add_security_risk_prediction: bool = False,
        on_token: AnyTokenCallbackType | None = None,
        call_context: "LLMCallContext | None" = None,
        **kwargs: Any,
    ) -> LLMResponse:
        response = self.completion(
            messages=messages,
            tools=tools,
            add_security_risk_prediction=add_security_risk_prediction,
            call_context=call_context,
            **kwargs,
        )
        return response


def test_state_persists_llm_response_metadata_sidecar(tmp_path):
    llm = TestLLM.from_messages([_message()])
    conversation = Conversation(
        agent=Agent(llm=llm, tools=[]),
        persistence_dir=str(tmp_path),
        workspace=str(tmp_path),
    )
    response = llm.completion([Message(role="user", content=[TextContent(text="hi")])])
    response = response.model_copy(update={"logprobs": _logprobs()})

    event_count = len(conversation.state.events)
    conversation.state.save_llm_response_metadata(response)

    assert len(conversation.state.events) == event_count
    metadata = conversation.state.load_llm_response_metadata(response.id)
    assert metadata is not None
    assert metadata.llm_response_id == response.id
    assert metadata.logprobs is not None
    assert metadata.logprobs.content[0].token == "done"

    assert conversation.state.persistence_dir is not None
    sidecar_path = Path(
        conversation.state.persistence_dir
    ) / llm_response_metadata_path(response.id)
    assert sidecar_path.exists()

    resumed = Conversation(
        agent=Agent(llm=TestLLM.from_messages([]), tools=[]),
        conversation_id=conversation.id,
        persistence_dir=str(tmp_path),
        workspace=str(tmp_path),
    )
    resumed_metadata = resumed.state.load_llm_response_metadata(response.id)
    assert resumed_metadata is not None
    assert resumed_metadata.logprobs is not None
    assert resumed_metadata.logprobs.content[0].token == "done"


def test_state_does_not_persist_empty_llm_response_metadata(tmp_path):
    llm = TestLLM.from_messages([_message()])
    conversation = Conversation(
        agent=Agent(llm=llm, tools=[]),
        persistence_dir=str(tmp_path),
        workspace=str(tmp_path),
    )
    response = llm.completion([Message(role="user", content=[TextContent(text="hi")])])

    conversation.state.save_llm_response_metadata(response)

    assert conversation.state.load_llm_response_metadata(response.id) is None
    assert conversation.state.persistence_dir is not None
    assert not (Path(conversation.state.persistence_dir) / LLM_RESPONSES_DIR).exists()


def test_agent_step_persists_llm_response_logprobs(tmp_path):
    llm = LogprobTestLLM.from_messages([_message()])
    conversation = Conversation(
        agent=Agent(llm=llm, tools=[]),
        persistence_dir=str(tmp_path),
        workspace=str(tmp_path),
    )

    conversation.send_message("say done")
    conversation.run()

    logprobs_by_response_id = conversation.state.load_llm_logprobs_by_response_id()
    assert logprobs_by_response_id["test-response-1"].content[0].token == "agent-token"

    agent_event = next(
        event
        for event in conversation.state.events
        if isinstance(event, MessageEvent) and event.source == "agent"
    )
    assert agent_event.llm_response_id in logprobs_by_response_id


@pytest.mark.asyncio
async def test_agent_astep_persists_llm_response_logprobs(tmp_path):
    llm = LogprobTestLLM.from_messages([_message()])
    conversation = Conversation(
        agent=Agent(llm=llm, tools=[]),
        persistence_dir=str(tmp_path),
        workspace=str(tmp_path),
    )

    conversation.send_message("say done")
    await conversation.arun()

    logprobs_by_response_id = conversation.state.load_llm_logprobs_by_response_id()
    assert logprobs_by_response_id["test-response-1"].content[0].token == "agent-token"
