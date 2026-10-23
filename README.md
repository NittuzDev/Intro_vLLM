# vLLM test and benchmarks

This is the environment used to run [`IntrovLLM.ipynb`](IntrovLLM.ipynb).

The notebook talks to two local servers over HTTP. vLLM listens on port 8000. Ollama listens on port 11434. Only one of them should have the model loaded on the GPU at a time.

## Tested environment

| | |
| --- | --- |
| OS | Red Hat Enterprise Linux 10.1 |
| GPU | NVIDIA L4, 25 GB VRAM |
| Python | 3.12.12 |
| vLLM | 0.32.0 |
| Ollama | 0.40.8 |
| Model | `Qwen/Qwen3-1.7B` in bfloat16, and the Ollama tag `qwen3:1.7b-fp16` |

`nvidia-smi` should already show the L4 before you start. These steps do not install the NVIDIA driver.

The Ollama tag `qwen3:1.7b` with no suffix is a 4-bit quant. The notebook compares vLLM's bfloat16 checkpoint with `qwen3:1.7b-fp16`.

## 1. Install vLLM

In the first terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv ~/vllm-env --python 3.12.12
source ~/vllm-env/bin/activate
uv pip install vllm==0.32.0 --torch-backend=auto
vllm download-kernels
```

`vllm download-kernels` fetches the precompiled kernels so the first server start does not compile them. Leave this environment with `deactivate`.

Do not start the server yet if you still need to download the Ollama model. Both servers want the same GPU.

## 2. Install Ollama

In a second terminal. Ollama is a system program, not a package inside the vLLM virtualenv. On RHEL the installer unpacks a `.tar.zst`, so `zstd` has to be present first.

```bash
sudo dnf install zstd
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.8 sh
ollama --version
```

The installer starts a systemd service. That service uses `OLLAMA_NUM_PARALLEL=1`, which makes the concurrency section a queue. Download the model with that service, then stop it and start Ollama yourself:

```bash
ollama pull qwen3:1.7b-fp16
sudo systemctl stop ollama
```

Check the tag:

```bash
ollama list
```

You want `qwen3:1.7b-fp16`. After a reboot, stop the service again before the command in the next section. Otherwise the one-slot service takes port 11434.

## 3. Notebook packages

In a third terminal:

```bash
git clone https://github.com/NittuzDev/Intro_vLLM.git
cd Intro_vLLM

uv venv ~/notebook-env --python 3.12.12
source ~/notebook-env/bin/activate
uv pip install -r requirements.txt jupyter
jupyter notebook IntrovLLM.ipynb
```

The notebook only needs `openai`, `requests`, and `matplotlib`. It does not import vLLM. Point the notebook kernel at `~/notebook-env`.

## 4. Run the notebook

Keep the notebook kernel running. Start and stop the servers around it.

**First request, logprobs, and the vLLM half of the comparison.** In the vLLM terminal:

```bash
source ~/vllm-env/bin/activate
vllm serve Qwen/Qwen3-1.7B --dtype=bfloat16 --max-model-len 4096
```

The first launch downloads the weights from Hugging Face into `~/.cache/huggingface/hub`. Wait until port 8000 answers, then run the notebook through the vLLM benchmark cell. That cell writes `outputs/vllm_vs_ollama.json`.

**Ollama half.** Stop `vllm serve` with Ctrl-C. vLLM keeps the weights on the GPU until that process exits. Then, in the Ollama terminal:

```bash
OLLAMA_NUM_PARALLEL=32 OLLAMA_CONTEXT_LENGTH=4096 ollama serve
```

Run the Ollama cell and the chart cell. Thirty-two slots of 4096 tokens need a large KV reservation on top of the fp16 weights. If the model cannot load, stop other GPU processes, or lower both `OLLAMA_NUM_PARALLEL` and `CONCURRENCY_LEVELS` in the notebook to the same maximum.

**Prefix caching.** Unload Ollama so vLLM can use the GPU.

```bash
ollama stop qwen3:1.7b-fp16
```

Start vLLM with prefix caching off:

```bash
vllm serve Qwen/Qwen3-1.7B --dtype=bfloat16 --max-model-len 4096 --no-enable-prefix-caching
```

Run the cache-off cell. It writes `outputs/prefix_caching_off.json`. Stop that server, then start the same model with caching on:

```bash
vllm serve Qwen/Qwen3-1.7B --dtype=bfloat16 --max-model-len 4096
```

Run the next code cell. It repeats the saved prompt and draws both runs.
