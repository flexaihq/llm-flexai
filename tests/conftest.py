import pytest

MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {
            "id": "bge-m3",
            "object": "model",
            "category": "embedding",
            "supports": ["embeddings"],
            "supported_parameters": ["input", "encoding_format"],
        },
        {
            "id": "DeepSeek-V4-Flash-0731",
            "object": "model",
            "name": "DeepSeek V4 Flash",
            "context_length": 131072,
            "category": "text",
            "supports": [
                "chat",
                "streaming",
                "tool_use",
                "reasoning_effort",
                "structured_outputs",
                "reasoning",
            ],
            "supported_parameters": ["tools", "response_format"],
        },
        {
            "id": "Llama-3.3-70B-Instruct-FP8",
            "object": "model",
            "category": "text",
            "supports": ["chat", "completion", "streaming", "tool_use"],
            "supported_parameters": ["tools"],
        },
        {
            "id": "Qwen3.6-27B-FP8",
            "object": "model",
            "category": "vision",
            "supports": ["chat", "streaming", "tool_use", "vision"],
            "supported_parameters": ["tools"],
        },
        {
            "id": "example-unverified-vision",
            "object": "model",
            "category": "vision",
            "supports": ["chat", "streaming", "tool_use", "vision"],
            "supported_parameters": ["tools"],
        },
        {
            "id": "example-ocr",
            "object": "model",
            "category": "vision",
            "supports": ["image_input", "structured_outputs"],
            "supported_parameters": [],
        },
        {
            "id": "FLUX.1-schnell",
            "object": "model",
            "category": "image",
            "supports": ["image_generation"],
            "supported_parameters": [],
        },
    ],
}

MODELS_URL = "https://api.flex.ai/v1/models"


@pytest.fixture
def user_path(tmp_path):
    path = tmp_path / "llm-user-dir"
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def no_network(httpx_mock, httpx2_mock):
    # Every test runs with both HTTP stacks mocked: the plugin fetches the
    # model list with httpx, and recent openai clients send requests with
    # httpx2. Any request without a registered response fails the test.
    return httpx_mock, httpx2_mock


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    monkeypatch.setenv("FLEXAI_API_KEY", "test-flexai-key")


@pytest.fixture
def mock_models(httpx_mock):
    httpx_mock.add_response(url=MODELS_URL, json=MODELS_RESPONSE)
    return httpx_mock
