# llm-flexai

[![PyPI](https://img.shields.io/pypi/v/llm-flexai.svg)](https://pypi.org/project/llm-flexai/)
[![Tests](https://github.com/flexaihq/llm-flexai/actions/workflows/test.yml/badge.svg)](https://github.com/flexaihq/llm-flexai/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flexaihq/llm-flexai/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for models hosted by [FlexAI](https://flex.ai/).

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-flexai
```

## Configuration

You will need an API key from [platform.flex.ai](https://platform.flex.ai). Set it like this:

```bash
llm keys set flexai
# Paste your key here
```

You can also set the `FLEXAI_API_KEY` environment variable instead.

## Usage

The list of models is fetched live from `GET https://api.flex.ai/v1/models` and cached for an hour. Every chat model is available with a `flexai/` prefix. To see them:

```bash
llm models -q flexai/
```

Run a prompt:

```bash
llm -m flexai/DeepSeek-V4-Flash-0731 "Five fun names for a pet pelican"
```

Start a chat:

```bash
llm chat -m flexai/gpt-oss-120b
```

### Tools

FlexAI chat models support [tool calling](https://llm.datasette.io/en/stable/tools.html):

```bash
llm -m flexai/DeepSeek-V4-Flash-0731 -T llm_version "What version of LLM is this?" --td
```

### Images

Models where image input has been verified accept image attachments (PNG, JPEG, WebP):

```bash
llm -m flexai/Qwen3.6-27B-FP8 "Describe this image" -a photo.jpg
```

The verified models are listed in `VISION_MODELS` in [llm_flexai.py](llm_flexai.py).

### Options

Models that support it take a `reasoning_effort` option:

```bash
llm -m flexai/gpt-oss-120b -o reasoning_effort low "Explain RAID 5 in one paragraph"
```

Models that support structured outputs work with `--schema`:

```bash
llm -m flexai/DeepSeek-V4-Flash-0731 --schema 'name, age int' "Invent a dog"
```

### Embeddings

The `bge-m3` embedding model is available as `flexai/bge-m3`:

```bash
llm embed -m flexai/bge-m3 -c "hello world"
```

### Commands

```bash
llm flexai models          # list FlexAI models and their capabilities
llm flexai models --json   # the raw /v1/models response
llm flexai refresh         # refresh the cached model list
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-flexai
python -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
pip install -e '.[test]'
```

To run the tests:

```bash
pytest
```

The tests mock every HTTP request and make no network calls.

## About FlexAI

FlexAI runs open models behind an OpenAI-compatible API at `https://api.flex.ai/v1`. Documentation: [docs.flex.ai](https://docs.flex.ai/inference-api/quickstart).
