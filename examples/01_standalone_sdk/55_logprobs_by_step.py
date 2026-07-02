"""Capture per-step LLM logprobs from a local agent conversation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LocalConversation,
    Tool,
)
from openhands.sdk.event import ActionEvent
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.llm.llm_response import LLMLogprobs
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


def response_id_from_event(event: Event) -> str | None:
    if isinstance(event, ActionEvent):
        return event.llm_response_id
    if isinstance(event, MessageEvent) and event.source == "agent":
        return event.llm_response_id
    return None


def collect_logprobs_by_step(
    conversation: LocalConversation,
    logprobs_by_response_id: dict[str, LLMLogprobs],
) -> dict[int, LLMLogprobs]:
    seen_response_ids: set[str] = set()
    logprobs_by_step: dict[int, LLMLogprobs] = {}

    for event in conversation.state.events:
        response_id = response_id_from_event(event)
        if response_id is None or response_id in seen_response_ids:
            continue
        seen_response_ids.add(response_id)
        logprobs = logprobs_by_response_id.get(response_id)
        if logprobs is not None:
            logprobs_by_step[len(seen_response_ids) - 1] = logprobs

    return logprobs_by_step


api_key = os.getenv("NRP_LLM_API_KEY") or os.getenv("LLM_API_KEY")
assert api_key is not None, "Set NRP_LLM_API_KEY or LLM_API_KEY."

model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL", "openai/qwen3-small")
base_url = os.getenv("LLM_BASE_URL")
if base_url is None and os.getenv("NRP_LLM_API_KEY"):
    base_url = "https://ellm.nrp-nautilus.io/v1"

llm = LLM(
    usage_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
    logprobs=True,
    top_logprobs=5,
)

tools = [
    Tool(name=TerminalTool.name),
    Tool(name=FileEditorTool.name),
]
agent = Agent(llm=llm, tools=tools)

with tempfile.TemporaryDirectory() as workspace:
    workspace_path = Path(workspace)
    conversation = Conversation(
        agent=agent,
        workspace=workspace_path,
        max_iteration_per_run=2,
        persistence_dir=Path(".agent_tmp/logprobs_example/no_bytes"),
    )

    conversation.send_message(
        "Briefly state the bash command you will run, then use bash to get the "
        "current system datetime and write it to hello.txt. After the file is "
        "written, tell me the exact command you used."
    )
    conversation.run()

    logprobs_by_step: dict[int, LLMLogprobs] = collect_logprobs_by_step(
        conversation, conversation.state.load_llm_logprobs_by_response_id()
    )

    print(f"Workspace: {workspace_path}")
    print(f"hello.txt exists: {(workspace_path / 'hello.txt').exists()}")
    print(f"Captured logprobs for steps: {sorted(logprobs_by_step)}")
    for step_index, logprobs in logprobs_by_step.items():
        preview = [(token.token, token.logprob) for token in logprobs.content[:10]]
        print(f"Step {step_index}: {preview}")

cost = llm.metrics.accumulated_cost
print(f"EXAMPLE_COST: {cost}")
