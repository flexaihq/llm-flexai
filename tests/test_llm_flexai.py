import json
import os
import time

import llm
from click.testing import CliRunner
from llm.cli import cli

import llm_flexai
from conftest import MODELS_RESPONSE, MODELS_URL

CHAT_URL = "https://api.flex.ai/v1/chat/completions"
EMBEDDINGS_URL = "https://api.flex.ai/v1/embeddings"


def flexai_model_ids():
    return sorted(
        model.model_id
        for model in llm.get_models()
        if model.model_id.startswith("flexai/")
    )


def chat_completion(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def test_no_key_registers_nothing(monkeypatch, httpx_mock):
    monkeypatch.delenv("FLEXAI_API_KEY")
    assert flexai_model_ids() == []
    assert httpx_mock.get_requests() == []


def test_registers_chat_models_only(mock_models):
    assert flexai_model_ids() == [
        "flexai/DeepSeek-V4-Flash-0731",
        "flexai/Llama-3.3-70B-Instruct-FP8",
        "flexai/Qwen3.6-27B-FP8",
        "flexai/example-unverified-vision",
    ]
    request = mock_models.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer test-flexai-key"


def test_model_flags(mock_models):
    model = llm.get_model("flexai/DeepSeek-V4-Flash-0731")
    assert isinstance(model, llm_flexai.FlexAIChat)
    assert str(model) == "FlexAI: flexai/DeepSeek-V4-Flash-0731"
    assert model.model_name == "DeepSeek-V4-Flash-0731"
    assert model.api_base == "https://api.flex.ai/v1"
    assert model.needs_key == "flexai"
    assert model.key_env_var == "FLEXAI_API_KEY"
    assert model.supports_tools is True
    assert model.supports_schema is True
    assert "reasoning_effort" in model.Options.model_fields
    assert not model.attachment_types

    plain = llm.get_model("flexai/Llama-3.3-70B-Instruct-FP8")
    assert plain.supports_schema is False
    assert "reasoning_effort" not in plain.Options.model_fields

    assert isinstance(
        llm.get_async_model("flexai/DeepSeek-V4-Flash-0731"),
        llm_flexai.FlexAIAsyncChat,
    )


def test_vision_only_for_verified_models(mock_models):
    verified = llm.get_model("flexai/Qwen3.6-27B-FP8")
    assert "Qwen3.6-27B-FP8" in llm_flexai.VISION_MODELS
    assert verified.attachment_types == set(llm_flexai.IMAGE_ATTACHMENT_TYPES)
    assert "application/pdf" not in verified.attachment_types

    unverified = llm.get_model("flexai/example-unverified-vision")
    assert "example-unverified-vision" not in llm_flexai.VISION_MODELS
    assert not unverified.attachment_types


def test_prompt(mock_models, httpx2_mock):
    httpx2_mock.add_response(
        url=CHAT_URL, method="POST", json=chat_completion(content="ok")
    )
    model = llm.get_model("flexai/DeepSeek-V4-Flash-0731")
    response = model.prompt("say ok", stream=False)
    assert response.text() == "ok"
    request = httpx2_mock.get_requests(url=CHAT_URL)[0]
    assert request.headers["Authorization"] == "Bearer test-flexai-key"
    body = json.loads(request.content)
    assert body["model"] == "DeepSeek-V4-Flash-0731"
    assert body["messages"] == [{"role": "user", "content": "say ok"}]
    assert "n" not in body
    assert "verbosity" not in body


def test_tool_call_round_trip(mock_models, httpx2_mock):
    httpx2_mock.add_response(
        url=CHAT_URL,
        method="POST",
        json=chat_completion(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }
            ]
        ),
    )
    httpx2_mock.add_response(
        url=CHAT_URL, method="POST", json=chat_completion(content="2 + 3 = 5")
    )
    calls = []

    def add(a: int, b: int) -> int:
        "Add two integers"
        calls.append((a, b))
        return a + b

    model = llm.get_model("flexai/DeepSeek-V4-Flash-0731")
    chain = model.chain("What is 2 + 3?", tools=[add], stream=False)
    assert chain.text() == "2 + 3 = 5"
    assert calls == [(2, 3)]

    first, second = [
        json.loads(r.content) for r in httpx2_mock.get_requests(url=CHAT_URL)
    ]
    assert first["tools"][0]["function"]["name"] == "add"
    tool_message = second["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"] == "5"


def test_embeddings(mock_models, httpx2_mock):
    httpx2_mock.add_response(
        url=EMBEDDINGS_URL,
        method="POST",
        json={
            "object": "list",
            "model": "bge-m3",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )
    model = llm.get_embedding_model("flexai/bge-m3")
    assert model.embed("hello") == [0.5, 0.25]
    body = json.loads(httpx2_mock.get_requests(url=EMBEDDINGS_URL)[0].content)
    assert body["model"] == "bge-m3"
    assert "dimensions" not in body


def test_models_cached(mock_models, user_path):
    flexai_model_ids()
    flexai_model_ids()
    assert len(mock_models.get_requests(url=MODELS_URL)) == 1
    cached = json.loads((user_path / "flexai_models.json").read_text())
    assert cached == MODELS_RESPONSE


def test_stale_cache_used_when_api_fails(httpx_mock, user_path):
    cache = user_path / "flexai_models.json"
    cache.write_text(json.dumps(MODELS_RESPONSE))
    old = time.time() - llm_flexai.CACHE_TIMEOUT - 60
    os.utime(cache, (old, old))
    httpx_mock.add_response(url=MODELS_URL, status_code=503)
    assert "flexai/DeepSeek-V4-Flash-0731" in flexai_model_ids()


def test_api_failure_without_cache_registers_nothing(httpx_mock):
    httpx_mock.add_response(url=MODELS_URL, status_code=401)
    assert flexai_model_ids() == []


def test_models_command(mock_models):
    result = CliRunner().invoke(cli, ["flexai", "models"])
    assert result.exit_code == 0, result.output
    assert "- id: flexai/DeepSeek-V4-Flash-0731" in result.output
    assert "context_length: 131,072" in result.output


def test_models_command_without_key(monkeypatch, httpx_mock):
    monkeypatch.delenv("FLEXAI_API_KEY")
    result = CliRunner().invoke(cli, ["flexai", "models"])
    assert result.exit_code == 1
    assert "llm keys set flexai" in result.output
