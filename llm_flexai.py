import json
import time
from pathlib import Path

import click
import httpx
import llm
from llm.default_plugins.openai_models import AsyncChat, Chat

API_BASE = "https://api.flex.ai/v1"
KEY_ALIAS = "flexai"
KEY_ENV_VAR = "FLEXAI_API_KEY"
MODEL_PREFIX = "flexai/"
CACHE_FILENAME = "flexai_models.json"
CACHE_TIMEOUT = 3600

# Image input has been verified end-to-end against these models. Other models
# may advertise vision in /v1/models, but they only accept attachments once
# they have been checked and added here.
VISION_MODELS = frozenset(
    {
        "DeepSeek-V4.1-Flash",
        "gemma-4-26B-A4B-it",
        "gemma-4-31b-it",
        "GLM-5.3-Flash",
        "Muse-Glimmer-30B",
        "Qwen3.5-9B",
        "Qwen3.6-27B-FP8",
        "Qwen3.6-35B-A3B-FP8",
        "Qwen3.8-27B",
        "Qwen3.8-Flash-Next",
        "Step-3.7-Flash",
    }
)

# The base class also accepts PDFs for vision models; FlexAI takes images only.
IMAGE_ATTACHMENT_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


def get_api_key():
    return llm.get_key(alias=KEY_ALIAS, env=KEY_ENV_VAR)


def _cache_path():
    return llm.user_dir() / CACHE_FILENAME


def fetch_models(key, skip_cache=False):
    """Return the model definitions from GET /v1/models.

    Results are cached in the LLM user directory for an hour. If the API
    cannot be reached, a stale cache is used; with no cache, an empty list is
    returned so a network problem never breaks the rest of the llm CLI.
    """
    path = Path(_cache_path())
    if not skip_cache and path.is_file():
        if time.time() - path.stat().st_mtime < CACHE_TIMEOUT:
            try:
                return json.loads(path.read_text())["data"]
            except (ValueError, KeyError):
                pass
    try:
        response = httpx.get(
            f"{API_BASE}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        data["data"]  # validate shape before caching
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data["data"]
    except (httpx.HTTPError, ValueError, KeyError):
        if path.is_file():
            try:
                return json.loads(path.read_text())["data"]
            except (ValueError, KeyError):
                return []
        return []


def _supports(model_definition):
    return set(model_definition.get("supports") or [])


def is_chat_model(model_definition):
    return "chat" in _supports(model_definition)


def is_embedding_model(model_definition):
    return "embeddings" in _supports(model_definition)


def supports_tools(model_definition):
    return "tool_use" in _supports(model_definition) or "tools" in (
        model_definition.get("supported_parameters") or []
    )


def chat_model_kwargs(model_definition):
    model_name = model_definition["id"]
    supports = _supports(model_definition)
    return dict(
        model_id=MODEL_PREFIX + model_name,
        model_name=model_name,
        api_base=API_BASE,
        vision=model_name in VISION_MODELS,
        reasoning="reasoning_effort" in supports,
        supports_schema="structured_outputs" in supports,
        supports_tools=supports_tools(model_definition),
    )


class _FlexAIMixin:
    needs_key = KEY_ALIAS
    key_env_var = KEY_ENV_VAR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.vision:
            self.attachment_types = set(IMAGE_ATTACHMENT_TYPES)

    def __str__(self):
        return "FlexAI: {}".format(self.model_id)


class FlexAIChat(_FlexAIMixin, Chat):
    pass


class FlexAIAsyncChat(_FlexAIMixin, AsyncChat):
    pass


class FlexAIEmbeddingModel(llm.EmbeddingModel):
    needs_key = KEY_ALIAS
    key_env_var = KEY_ENV_VAR
    batch_size = 32

    def __init__(self, model_name):
        self.model_id = MODEL_PREFIX + model_name
        self.model_name = model_name

    def embed_batch(self, items, *, key=None):
        import openai

        client = openai.OpenAI(api_key=key, base_url=API_BASE)
        results = client.embeddings.create(
            input=list(items), model=self.model_name
        ).data
        return ([float(r) for r in result.embedding] for result in results)


@llm.hookimpl
def register_models(register):
    key = get_api_key()
    if not key:
        return
    for model_definition in fetch_models(key):
        if not is_chat_model(model_definition):
            continue
        kwargs = chat_model_kwargs(model_definition)
        register(FlexAIChat(**kwargs), FlexAIAsyncChat(**kwargs))


@llm.hookimpl
def register_embedding_models(register):
    key = get_api_key()
    if not key:
        return
    for model_definition in fetch_models(key):
        if is_embedding_model(model_definition):
            register(FlexAIEmbeddingModel(model_definition["id"]))


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def flexai():
        "Commands relating to the llm-flexai plugin"

    @flexai.command()
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(json_):
        "List models available from FlexAI"
        key = get_api_key()
        if not key:
            raise click.ClickException(
                "No key found - run 'llm keys set flexai' or set FLEXAI_API_KEY"
            )
        all_models = fetch_models(key)
        if json_:
            click.echo(json.dumps(all_models, indent=2))
            return
        for model in all_models:
            bits = [f"- id: {MODEL_PREFIX}{model['id']}"]
            if model.get("name"):
                bits.append(f"  name: {model['name']}")
            if model.get("context_length"):
                bits.append(f"  context_length: {model['context_length']:,}")
            if model.get("supports"):
                bits.append("  supports: " + ", ".join(model["supports"]))
            click.echo("\n".join(bits) + "\n")

    @flexai.command()
    def refresh():
        "Refresh the cached list of FlexAI models"
        key = get_api_key()
        if not key:
            raise click.ClickException(
                "No key found - run 'llm keys set flexai' or set FLEXAI_API_KEY"
            )
        before = {m["id"] for m in fetch_models(key)}
        after = {m["id"] for m in fetch_models(key, skip_cache=True)}
        added = sorted(after - before)
        removed = sorted(before - after)
        if added:
            click.echo(
                "Added models: " + ", ".join(MODEL_PREFIX + m for m in added),
                err=True,
            )
        if removed:
            click.echo(
                "Removed models: " + ", ".join(MODEL_PREFIX + m for m in removed),
                err=True,
            )
        if not added and not removed:
            click.echo("No changes", err=True)
