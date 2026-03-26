<p align="center">
  <img src="assets/logo.png" alt="MolmoWeb" width="100%">
</p>

<p align="center">
  <a href="https://allenai.org/papers/molmoweb">Paper</a> &nbsp;|&nbsp;
  <a href="https://allenai.org/blog/molmoweb">Blog Post</a> &nbsp;|&nbsp;
  <a href="https://molmoweb.allen.ai">Demo</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/collections/allenai/molmoweb">Models</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/collections/allenai/molmoweb-data">Data</a>
</p>

---

**MolmoWeb** is an open multimodal web agent built by [Ai2](https://allenai.org). Given a natural-language task, MolmoWeb autonomously controls a web browser -- clicking, typing, scrolling, and navigating -- to complete the task. This repository contains the agent code, inference client, evaluation benchmarks, and everything needed to reproduce the results from the paper.

## Table of Contents

- [Models](#models)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Download the Model](#1-download-the-model)
  - [Start the Model Server](#2-start-the-model-server)
  - [Test the Model](#3-test-the-model)
- [Inference Client](#inference-client)
  - [Single Query](#single-query)
  - [Batch Queries](#batch-queries)
  - [Extract Accessibility Tree](#extract-accessibility-tree)
- [License](#license)
- [TODO](#todo)

---

## Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| MolmoWeb-8B | 8B | [allenai/MolmoWeb-8B](https://huggingface.co/allenai/MolmoWeb-8B) |
| MolmoWeb-4B | 4B | [allenai/MolmoWeb-4B](https://huggingface.co/allenai/MolmoWeb-4B) |
| MolmoWeb-8B-Native | 8B | [allenai/MolmoWeb-8B-Native](https://huggingface.co/allenai/MolmoWeb-8B-Native) |
| MolmoWeb-4B-Native | 4B | [allenai/MolmoWeb-4B-Native](https://huggingface.co/allenai/MolmoWeb-4B-Native) |

The first two models (MolmoWeb-8B and MolmoWeb-4B) are Huggingface/transformers-compatible (see [example usage](https://huggingface.co/allenai/MolmoWeb-8B#quick-start) on Huggingface); and the last two (MolmoWeb-8B-Native and MolmoWeb-4B-Native) are molmo-native checkpoints. 

**Collections:**
- [All MolmoWeb Models](https://huggingface.co/collections/allenai/molmoweb)
- [MolmoWeb Training Data](https://huggingface.co/collections/allenai/molmoweb-data)

---

## Installation

Requires Python 3.10+. We use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone git@github.com:allenai/molmoweb.git
cd molmoweb
uv venv
uv sync

# Install Playwright browsers (needed for local browser control)
uv run playwright install
uv run playwright install --with-deps chromium
```

---

### Environment Variables

```bash
# Browserbase (required when --env_type browserbase)
export BROWSERBASE_API_KEY="your-browserbase-api-key"
export BROWSERBASE_PROJECT_ID="your-browserbase-project-id"

# Google Gemini (required for gemini_cua, gemini_axtree, and Gemini-based judges)
export GOOGLE_API_KEY="your-google-api-key"

# OpenAI (required for gpt_axtree and GPT-based judges like webvoyager)
export OPENAI_API_KEY="your-openai-api-key"
```

---

## Quick Start

Three helper scripts in `scripts/` let you download weights, start the server, and test it end-to-end.

### 1. Download the Model

```bash
bash scripts/download_weights.sh                                  # MolmoWeb-8B (default)
bash scripts/download_weights.sh allenai/MolmoWeb-4B-Native       # MolmoWeb-4B Native
```

This downloads the weights to `./checkpoints/<model-name>`.

### 2. Start the Model Server

```bash
bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B              # MolmoWeb-8B, port 8001
bash scripts/start_server.sh ./checkpoints/MolmoWeb-4B-Native       # MolmoWeb-4B
bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B 8002         # custom port
```

Or configure via environment variables:

```bash
export CKPT="./checkpoints/MolmoWeb-4B-Native"   # local path to downloaded weights
export PREDICTOR_TYPE="native"             # "native" or "hf"
export NUM_PREDICTORS=1                    # number of GPU workers
export DEVICE="mps"                       # optional: force device (mps/cpu/cuda:0)

bash scripts/start_server.sh
```

The server exposes a single endpoint:

```
POST http://127.0.0.1:8001/predict
{
  "prompt": "...",
  "image_base64": "..."
}
```

Wait for the server to print that the model is loaded, then test it.

### 3. Test the Model

Once the server is running, send it a screenshot of the [Ai2 careers page](https://allenai.org/careers) (included in `assets/test_screenshot.png`) and ask it to read the job titles:

```bash
uv run python scripts/test_server.py                        # default: localhost:8001
uv run python scripts/test_server.py http://myhost:8002     # custom endpoint
```

The test script sends this prompt to the model:

> Read the text on this page. What are the first four job titles listed under 'Open roles'?

You can also do it manually in a few lines of Python:

```python
import base64, requests

with open("assets/test_screenshot.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://127.0.0.1:8001/predict", json={
    "prompt": "What are the first four job titles listed under 'Open roles'?",
    "image_base64": image_b64,
})
print(resp.json())
```

---

## Inference Client

The `inference` package provides a high-level Python client that manages a browser session and runs the agent end-to-end. The client communicates with a running model server endpoint.

### Single Query

```python
from inference import MolmoWeb

client = MolmoWeb(
    endpoint="SET_UP_YOUR_ENDPOINT",
    local=True,         # True = local Chromium, False = Browserbase cloud browser
    headless=True,
) 

query = "Go to arxiv.org and find out the paper about Molmo and Pixmo."
traj = client.run(query=query, max_steps=10)

output_path = traj.save_html(query=query)
print(f"Saved to {output_path}")
```

### Follow-up Query

```python
followup_query = "Find the full author list of the paper."
traj2 = client.continue_run(query=followup_query, max_steps=10)
```

### Batch Queries

```python
queries = [
    "Go to allenai.org and find the latest research papers on top of the homepage",
    "Search for 'OLMo' on Wikipedia",
    "What is the weather in Seattle today?",
]

trajectories = client.run_batch(
    queries=queries,
    max_steps=10,
    max_workers=3,
) # Inspect the trajectory .html files default saved under inference/htmls
```

### Inference Backends

Supported backends: `fastapi` (remote HTTP endpoint), `modal` (serverless), `native` (native molmo/olmo-compatible checkpoint), `hf` (HuggingFace Transformers-compatible checkpoint).

> **vLLM support coming soon.**

### Extract Accessibility Tree

```
from inference.client import MolmoWeb

client = MolmoWeb()
axtree_str = client.get_axtree("https://allenai.org/")
print(axtree_str)
client.close()
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## デフォルト実装からの変更点（ローカル）

この作業ツリーでは、デフォルトの MolmoWeb コードに対して主に以下を変更しています。

- `agent/fastapi_model_server.py`
  - `DEVICE` 環境変数によるデバイス明示指定（例: `mps` / `cpu` / `cuda:0`）に対応。
  - `CKPT` 未指定時に明示的にエラーを出すように変更。
  - Predictor のデバイス割り当てを `cuda` / `mps` / `cpu` の可用性に応じて自動化。
- `agent/model_backends.py`
  - HF / Native バックエンドで `mps` / `cpu` を含むデバイス自動選択を追加。
  - デバイスに応じた dtype（`bfloat16` / `float16` / `float32`）の切り替えを追加。
  - Native 側で `mps` 利用時のキャッシュクリア処理を追加。
- `scripts/start_server.sh`
  - `DEVICE` の説明と起動時ログ表示を追加（デバイス上書き確認用）。

<!--
### macOS に未対応だった理由と、対応できた変更点

- 未対応だった主因
  - 既定のデバイス前提が CUDA 寄りで、GPU が見つからない環境（Apple Silicon の `mps` / CPU）へのフォールバックが弱かった。
  - 一部実装でデバイス・dtype の選択ロジックが固定的で、CUDA 前提の設定がそのまま使われやすかった。
  - 起動スクリプト側で「どのデバイスで動かすか」を明示指定しづらく、macOS での運用パラメータが見えにくかった。

- 今回の対応内容（どう直したか）
  - サーバ起動時に `DEVICE` 環境変数を受け取り、`mps` / `cpu` / `cuda:0` を明示的に指定可能にした。
  - `DEVICE` 未指定時は、`cuda` → `mps` → `cpu` の順で可用性を判定して自動選択するようにした。
  - HF バックエンドではデバイス別に dtype を切り替え（CUDA=`bfloat16`, MPS=`float16`, CPU=`float32`）るようにした。
  - Native バックエンドでも同様に `mps`/`cpu` を選べるようにし、`mps` 利用時のキャッシュクリア処理も追加した。
  - `scripts/start_server.sh` に `DEVICE` の説明と起動時ログを追加し、macOS での設定確認を容易にした。

- 使い方（macOS の例）
  - `DEVICE=mps bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B`
  - `DEVICE=cpu bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B`

補足: ローカルには `checkpoints/` や `seimiya/` などの未追跡ファイルも存在します（実験用・作業用）。
-->
## TODO

- [x] Inference
- [ ] Eval
- [ ] Training
