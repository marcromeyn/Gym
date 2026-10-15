# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A sandbox operation carries the rollout and tool call it ran for."""

import asyncio
import contextvars
from urllib.parse import urlsplit

import orjson
import pytest
from fastapi import Body, FastAPI
from omegaconf import OmegaConf
from starlette.testclient import TestClient

import nemo_gym.server_utils
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.rollout_correlation import (
    TOOL_CALL_ID_HEADER,
    RolloutContextMiddleware,
    current_rollout_id,
    current_tool_call_id,
    rollout_context,
    tool_call_context,
)
from nemo_gym.sandbox import AsyncSandbox, Sandbox, SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.api import ROLLOUT_ID_METADATA_KEY, TOOL_CALL_ID_METADATA_KEY
from nemo_gym.server_utils import ServerClient, direct_http_tool_target, get_response_json
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.simple_agent.app import ModelServerRef, ResourcesServerRef, SimpleAgent, SimpleAgentConfig


class _RecordingProvider:
    name = "recording"

    def __init__(self) -> None:
        self.created_specs: list[SandboxSpec] = []

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.created_specs.append(spec)
        return SandboxHandle(sandbox_id="sbx-1", provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: object) -> SandboxExecResult:
        return SandboxExecResult(stdout=command, stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: object, target_path: str) -> None:
        pass

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: object) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        pass

    async def aclose(self) -> None:
        pass


class _PtySession:
    closed = False

    def __init__(self) -> None:
        self._out = [b"pty-out"]

    async def read(self, *, timeout_s: float | None = None) -> bytes:
        return self._out.pop(0) if self._out else b""

    async def read_stderr(self, *, timeout_s: float | None = None) -> bytes:
        return b""

    async def wait_exit(self, *, timeout_s: float | None = None) -> int:
        return 0

    async def close(self) -> None:
        self.closed = True


class _PtyProvider(_RecordingProvider):
    async def create_pty(self, handle: SandboxHandle, spec: object) -> _PtySession:
        return _PtySession()


async def _started(provider: _RecordingProvider | None = None, **spec_fields: object) -> AsyncSandbox:
    return await AsyncSandbox(provider or _RecordingProvider(), SandboxSpec(image="ubuntu", **spec_fields)).start()


async def test_outside_a_rollout_nothing_changes() -> None:
    provider = _RecordingProvider()
    spec = SandboxSpec(image="ubuntu", metadata={"team": "a"})
    sandbox = await AsyncSandbox(provider, spec).start()

    result = await sandbox.exec("echo hi")

    assert result == SandboxExecResult(stdout="echo hi", stderr=None, return_code=0)
    assert (result.rollout_id, result.tool_call_id) == (None, None)
    assert provider.created_specs == [spec]


async def test_a_tool_call_without_a_rollout_is_not_attributed() -> None:
    provider = _RecordingProvider()
    with tool_call_context("call_1"):
        sandbox = await _started(provider)
        result = await sandbox.exec("echo hi")

    assert (result.rollout_id, result.tool_call_id) == (None, None)
    assert provider.created_specs[0].metadata == {}


async def test_exec_result_carries_the_rollout_and_tool_call() -> None:
    sandbox = await _started()
    with rollout_context("4-2"):
        in_rollout = await sandbox.exec("ls")
        with tool_call_context("call_abc"):
            in_tool_call = await sandbox.exec("ls")

    assert (in_rollout.rollout_id, in_rollout.tool_call_id) == ("4-2", None)
    assert (in_tool_call.rollout_id, in_tool_call.tool_call_id) == ("4-2", "call_abc")
    assert (in_tool_call.stdout, in_tool_call.return_code) == ("ls", 0)


async def test_pty_exec_result_carries_the_rollout_and_tool_call() -> None:
    sandbox = await _started(_PtyProvider())
    with rollout_context("4-2"), tool_call_context("call_abc"):
        result = await sandbox.pty.exec("ls")

    assert (result.stdout, result.rollout_id, result.tool_call_id) == ("pty-out", "4-2", "call_abc")


async def test_tasks_inherit_the_context_they_were_created_in() -> None:
    sandbox = await _started()

    async def run(rollout_id: str, tool_call_id: str) -> SandboxExecResult:
        with rollout_context(rollout_id), tool_call_context(tool_call_id):
            return await asyncio.create_task(sandbox.exec("ls"))

    first, second = await asyncio.gather(run("1-0", "call_a"), run("2-0", "call_b"))

    assert (first.rollout_id, first.tool_call_id) == ("1-0", "call_a")
    assert (second.rollout_id, second.tool_call_id) == ("2-0", "call_b")


def test_sync_sandbox_carries_the_callers_context_to_its_loop_thread() -> None:
    """``Sandbox`` runs each call on a private event loop thread; the caller's context must cross it."""
    provider = _RecordingProvider()
    with Sandbox(provider, SandboxSpec(image="ubuntu")) as sandbox:
        with rollout_context("4-2"), tool_call_context("call_abc"):
            sandbox.start()
            result = sandbox.exec("ls")
        outside = sandbox.exec("ls")

    assert (result.rollout_id, result.tool_call_id) == ("4-2", "call_abc")
    assert (outside.rollout_id, outside.tool_call_id) == (None, None)
    assert provider.created_specs[0].metadata == {
        ROLLOUT_ID_METADATA_KEY: "4-2",
        TOOL_CALL_ID_METADATA_KEY: "call_abc",
    }


async def test_start_labels_the_sandbox_and_explicit_metadata_wins() -> None:
    provider = _RecordingProvider()
    with rollout_context("4-2-a1"), tool_call_context("call_abc"):
        await _started(provider, metadata={"team": "a", TOOL_CALL_ID_METADATA_KEY: "set-by-caller"})

    assert provider.created_specs[0].metadata == {
        ROLLOUT_ID_METADATA_KEY: "4-2-a1",
        TOOL_CALL_ID_METADATA_KEY: "set-by-caller",
        "team": "a",
    }


async def test_ids_that_are_not_label_values_are_left_out_of_labels_only() -> None:
    """A Kubernetes label value is at most 63 characters and ends alphanumerically. An id that
    does not fit is omitted rather than truncated, but spans and results still carry it whole."""
    long_rollout = "r" * 64
    provider = _RecordingProvider()
    with rollout_context(long_rollout), tool_call_context("call_ends_with_dash-"):
        sandbox = await _started(provider)
        result = await sandbox.exec("ls")

    assert provider.created_specs[0].metadata == {}
    assert (result.rollout_id, result.tool_call_id) == (long_rollout, "call_ends_with_dash-")


@pytest.mark.parametrize("tool_call_id", ["", "call 1", "call\r\nx", "-call", "c" * 257, 7])
def test_malformed_tool_call_ids_are_unknown(tool_call_id: object) -> None:
    with tool_call_context(tool_call_id):
        assert current_tool_call_id() is None


async def test_middleware_exposes_the_tool_call_only_inside_a_rollout() -> None:
    observed: list[tuple[str | None, str | None]] = []

    async def inner_app(scope, receive, send) -> None:
        observed.append((current_rollout_id(), current_tool_call_id()))

    middleware = RolloutContextMiddleware(inner_app)
    header = [(TOOL_CALL_ID_HEADER.encode(), b"call_abc")]
    await middleware({"type": "http", "path": "/ng-rollout/4-2/exec", "headers": header}, None, None)
    await middleware({"type": "http", "path": "/exec", "headers": header}, None, None)
    await middleware({"type": "http", "path": "/ng-rollout/4-2/exec", "headers": []}, None, None)

    assert observed == [("4-2", "call_abc"), (None, None), ("4-2", None)]
    assert current_tool_call_id() is None


async def test_middleware_can_keep_the_prefix_for_routes_that_read_it() -> None:
    """Agents keep the prefix for their ``/ng-rollout/{rollout_id}/v1/responses`` twins and still get the context."""
    observed: list[tuple[str, str | None, str | None]] = []

    async def inner_app(scope, receive, send) -> None:
        observed.append((scope["path"], current_rollout_id(), current_tool_call_id()))

    middleware = RolloutContextMiddleware(inner_app, strip_prefix=False)
    header = [(TOOL_CALL_ID_HEADER.encode(), b"call_abc")]
    await middleware({"type": "http", "path": "/ng-rollout/4-2/v1/responses", "headers": []}, None, None)
    await middleware(
        {"type": "http", "path": "/ng-rollout/4-2/training-token-capture/v1/responses", "headers": header}, None, None
    )
    await middleware({"type": "http", "path": "/v1/responses", "headers": header}, None, None)

    assert observed == [
        ("/ng-rollout/4-2/v1/responses", "4-2", None),
        ("/ng-rollout/4-2/training-token-capture/v1/responses", "4-2", "call_abc"),
        ("/v1/responses", None, None),
    ]


@pytest.mark.parametrize(
    ("config", "base_url", "expected_url", "expected_headers"),
    [
        # Observability or checkpointing attributes the call; the grant's own headers are kept.
        (
            {"observability_enabled": True},
            "http://tools:80",
            "http://tools:80/ng-rollout/4-2",
            {"X-Scoped": "t", TOOL_CALL_ID_HEADER: "call_abc"},
        ),
        (
            {"checkpoint": {"enabled": True, "control_auth_token": "secret"}},
            "http://tools:80/",
            "http://tools:80/ng-rollout/4-2",
            {"X-Scoped": "t", TOOL_CALL_ID_HEADER: "call_abc"},
        ),
        # A grant that already carries a rollout prefix is not prefixed again.
        (
            {"observability_enabled": True},
            "http://tools:80/ng-rollout/9-9",
            "http://tools:80/ng-rollout/9-9",
            {"X-Scoped": "t", TOOL_CALL_ID_HEADER: "call_abc"},
        ),
        ({"observability_enabled": False}, "http://tools:80", "http://tools:80", {"X-Scoped": "t"}),
    ],
)
def test_direct_http_tool_target(config, base_url, expected_url, expected_headers) -> None:
    grant_headers = {"X-Scoped": "t"}
    with rollout_context("4-2"), tool_call_context("call_abc"):
        target = direct_http_tool_target(base_url, grant_headers, config)

    assert target == (expected_url, expected_headers)
    assert grant_headers == {"X-Scoped": "t"}
    assert direct_http_tool_target(base_url, grant_headers, config) == (base_url, grant_headers)


class _Response:
    """The parts of an aiohttp ``ClientResponse`` the Agent and ``get_response_json`` read."""

    def __init__(self, status: int, content: bytes, cookies: object = None) -> None:
        self.status = status
        self.ok = status < 400
        self.cookies = cookies if cookies is not None else {}
        self._content = content
        self.content = self

    @classmethod
    def of(cls, response) -> "_Response":
        return cls(response.status_code, response.content, response.cookies)

    async def read(self) -> bytes:
        return self._content


def _call_in_fresh_context(client: TestClient, method: str, url: str, **kwargs):
    # A fresh context stands in for the Resources Server process: the in-process test client would
    # otherwise hand it this caller's context variables.
    return contextvars.Context().run(
        client.request,
        method,
        urlsplit(url).path,
        json=kwargs.get("json"),
        headers=kwargs.get("headers"),
    )


class _SandboxToolServer(SimpleResourcesServer):
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/run_command")(self.run_command)
        return app

    async def run_command(self, body: dict = Body()) -> dict:
        provider = _RecordingProvider()
        async with AsyncSandbox(provider, SandboxSpec(image="ubuntu")) as sandbox:
            await sandbox.start()
            result = await sandbox.exec(body["command"])
        return {
            "rollout_id": result.rollout_id,
            "tool_call_id": result.tool_call_id,
            "metadata": provider.created_specs[0].metadata,
        }

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        raise NotImplementedError


@pytest.mark.parametrize("observability_enabled", [True, False])
async def test_tool_call_reaches_a_sandbox_on_the_resources_server(monkeypatch, observability_enabled: bool) -> None:
    """The Agent's tool call id crosses the HTTP hop with the rollout prefix, and only with it."""
    config = OmegaConf.create(
        {
            "observability_enabled": observability_enabled,
            "resources": {"resources_servers": {"tools": {"host": "resources.test", "port": 80}}},
        }
    )
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80), global_config_dict=config
    )
    resources = _SandboxToolServer(
        config=BaseResourcesServerConfig(host="resources.test", port=80, entrypoint="app.py", name="resources"),
        server_client=server_client,
    )
    client = TestClient(resources.setup_webserver())
    sent_headers: list[dict] = []

    async def dispatch(method: str, url: str, **kwargs):
        sent_headers.append(dict(kwargs.get("headers") or {}))
        return _Response.of(_call_in_fresh_context(client, method, url, **kwargs))

    monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)

    with rollout_context("4-2"), tool_call_context("call_abc"):
        response = await server_client.post(server_name="resources", url_path="/run_command", json={"command": "ls"})
    body = await get_response_json(response)

    if observability_enabled:
        assert sent_headers == [{TOOL_CALL_ID_HEADER: "call_abc"}]
        assert body == {
            "rollout_id": "4-2",
            "tool_call_id": "call_abc",
            "metadata": {ROLLOUT_ID_METADATA_KEY: "4-2", TOOL_CALL_ID_METADATA_KEY: "call_abc"},
        }
    else:
        assert sent_headers == [{}]
        assert body == {"rollout_id": None, "tool_call_id": None, "metadata": {}}


def _model_turn(output: dict) -> bytes:
    return orjson.dumps(
        {
            "id": f"resp_{output['id']}",
            "created_at": 0.0,
            "model": "policy",
            "object": "response",
            "output": [output],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


class _AgentWorld:
    """A real Simple Agent and Resources Server joined by a fake network that records every hop."""

    def __init__(self, monkeypatch, *, observability_enabled: bool) -> None:
        config = OmegaConf.create(
            {
                "observability_enabled": observability_enabled,
                "model": {"responses_api_models": {"policy": {"host": "model.test", "port": 80}}},
                "resources": {"resources_servers": {"tools": {"host": "resources.test", "port": 80}}},
            }
        )
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="head.test", port=80), global_config_dict=config
        )
        resources = _SandboxToolServer(
            config=BaseResourcesServerConfig(host="resources.test", port=80, entrypoint="app.py", name="resources"),
            server_client=server_client,
        )
        self.resources_client = TestClient(resources.setup_webserver())
        agent = SimpleAgent(
            config=SimpleAgentConfig(
                host="agent.test",
                port=80,
                entrypoint="app.py",
                name="agent",
                model_server=ModelServerRef(type="responses_api_models", name="model"),
                resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            ),
            server_client=server_client,
        )
        self.agent_client = TestClient(agent.setup_webserver())
        self.model_turns = [
            {
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "run_command",
                "arguments": '{"command": "ls"}',
                "type": "function_call",
                "status": "completed",
            },
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ]
        self.model_paths: list[str] = []
        self.tool_calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(nemo_gym.server_utils, "request", self.dispatch)
        monkeypatch.setattr("responses_api_agents.simple_agent.app.http_request", self.dispatch)

    async def dispatch(self, method: str, url: str, **kwargs):
        parts = urlsplit(url)
        if parts.hostname == "model.test":
            self.model_paths.append(parts.path)
            return _Response(200, _model_turn(self.model_turns.pop(0)))
        self.tool_calls.append((parts.path, dict(kwargs.get("headers") or {})))
        return _Response.of(_call_in_fresh_context(self.resources_client, method, url, **kwargs))

    def tool_output(self, response) -> dict:
        assert response.status_code == 200, response.text
        outputs = [item for item in response.json()["output"] if item["type"] == "function_call_output"]
        assert [item["call_id"] for item in outputs] == ["call_abc"]
        return orjson.loads(outputs[0]["output"])


_ATTRIBUTED_SANDBOX = {
    "rollout_id": "4-2",
    "tool_call_id": "call_abc",
    "metadata": {ROLLOUT_ID_METADATA_KEY: "4-2", TOOL_CALL_ID_METADATA_KEY: "call_abc"},
}
_UNATTRIBUTED_SANDBOX = {"rollout_id": None, "tool_call_id": None, "metadata": {}}


@pytest.mark.parametrize("observability_enabled", [True, False])
async def test_rollout_prefixed_responses_attribute_their_tool_calls(monkeypatch, observability_enabled: bool) -> None:
    """A rollout-prefixed ``/v1/responses`` runs its whole tool loop in that rollout.

    The legacy ``/run`` self-call reaches the Agent as a new request, so the rollout context ``/run`` sets does
    not cross it: the prefix on the request is all the Agent has. Its tool call must then carry the prefix and the
    tool call id to the Resources Server, where a sandbox sees both.
    """
    world = _AgentWorld(monkeypatch, observability_enabled=observability_enabled)

    response = world.agent_client.post("/ng-rollout/4-2/v1/responses", json={"input": "list files"})

    assert world.model_paths == ["/ng-rollout/4-2/v1/responses"] * 2
    if observability_enabled:
        assert world.tool_calls == [("/ng-rollout/4-2/run_command", {TOOL_CALL_ID_HEADER: "call_abc"})]
        assert world.tool_output(response) == _ATTRIBUTED_SANDBOX
    else:
        assert world.tool_calls == [("/run_command", {})]
        assert world.tool_output(response) == _UNATTRIBUTED_SANDBOX


async def test_unprefixed_responses_do_not_attribute_their_tool_calls(monkeypatch) -> None:
    world = _AgentWorld(monkeypatch, observability_enabled=True)

    response = world.agent_client.post("/v1/responses", json={"input": "list files"})

    assert world.tool_calls == [("/run_command", {})]
    assert world.tool_output(response) == _UNATTRIBUTED_SANDBOX


@pytest.mark.parametrize("observability_enabled", [True, False])
async def test_direct_http_tool_calls_carry_the_rollout_and_tool_call(
    monkeypatch, observability_enabled: bool
) -> None:
    """An Environment Server episode reaches its Resources Server through a direct HTTP grant, not ServerClient.

    The grant names the Resources Server root, so the Agent applies the same prefix and header to it.
    """
    world = _AgentWorld(monkeypatch, observability_enabled=observability_enabled)
    seed = world.agent_client.post(
        "/v1/agent_sessions",
        json=AgentSeedSessionRequest(
            agent_session_id="agent-session",
            episode_id=EpisodeId(rollout_id="4-2", attempt=0),
            task_id=TaskId(taskset="example", task_id="0"),
            tool_accesses=[
                DirectHTTPToolAccess(
                    name="tools.direct_http",
                    required=True,
                    base_url="http://resources.test:80",
                    headers={"X-Scoped-Access": "token"},
                )
            ],
        ).model_dump(mode="json"),
    )
    assert seed.status_code == 200, seed.text

    response = world.agent_client.post("/ng-rollout/4-2/v1/responses", json={"input": "list files"})

    if observability_enabled:
        assert world.tool_calls == [
            ("/ng-rollout/4-2/run_command", {"X-Scoped-Access": "token", TOOL_CALL_ID_HEADER: "call_abc"})
        ]
        assert world.tool_output(response) == _ATTRIBUTED_SANDBOX
    else:
        assert world.tool_calls == [("/run_command", {"X-Scoped-Access": "token"})]
        assert world.tool_output(response) == _UNATTRIBUTED_SANDBOX
