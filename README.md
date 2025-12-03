# Dria Benchmark System

A comprehensive benchmarking platform for comparing LLM (Large Language Model) performance across different GPU configurations and inference engines. Supports multiple cloud infrastructure providers including RunPod, Modal, Lightning AI, and Scaleway.

## 🎯 Project Aim

We aim to show and allow users to compare performances of different LLM models across various GPU configurations and inference engines. The system currently supports:

- **Cloud Provider**: RunPod, Modal, Lightning AI, Scaleway
- **Inference Engines**: Ollama, SGLang, vLLM, TensorRT-LLM, LMStudio, MLX-lm
- **Deployment**: Same LLM models with different inference engines on the same GPU for fair performance comparison

## 🌐 View Benchmark Results

**📊 [Visit dria.co/inference-benchmark](https://dria.co/inference-benchmark) to view and compare benchmark results in real-time!**

The web platform provides:
- Interactive benchmark comparisons
- Real-time GPU pricing
- Performance analytics and insights
- Community discussions and comments
- AI-powered recommendations

## 📊 Benchmarking Methodology

### Overview

After setting up a server on RunPod, we collect comprehensive data including:
- Server setup time
- LLM upload time
- Model loading time
- Benchmark execution time
- Performance metrics

We use an extended version of GuideLLM (customized for different inference engines) to create comprehensive benchmarks.

### Benchmark Types

#### 1. Concurrent Benchmark
- **Purpose**: Tests fixed concurrency levels
- **Range**: Rate 1 to 9 concurrent requests
- **Description**: Runs a fixed number of streams of requests in parallel
- **Usage**: `--rate` must be set to the desired concurrency level/number of streams

#### 2. Throughput Benchmark
- **Purpose**: Measures maximum processing capacity
- **Description**: A special test type designed to measure the maximum processing capacity of an LLM inference engine
- **Metrics**: 
  - Requests per second (RPS)
  - Maximum token generation speed (tokens per second - TPS)

### Benchmark Process

For each benchmark, we create **7 different benchmarks**:
1. **1 Throughput benchmark** (maximum capacity test)
2. **6 Concurrent benchmarks** (rates 1-6)

All 7 benchmarks are recorded and displayed to users for comprehensive performance analysis.

## 📈 Example Benchmark Data

Here's an example of the benchmark data structure and metrics collected:

| benchmark_type | rate | max_number | warmup_number | benchmark_duration | total_requests | successful_requests | requests_per_second | request_concurrency | request_latency | prompt_token_count | output_token_count | time_to_first_token_ms | time_per_output_token_ms | inter_token_latency_ms | output_tokens_per_second | tokens_per_second |
|----------------|------|------------|---------------|-------------------|----------------|---------------------|-------------------|-------------------|-----------------|-------------------|-------------------|------------------------|-------------------------|----------------------|------------------------|------------------|
| concurrent | 1 | - | - | 33.31 | 13 | 12 | 0.51 | 0.99 | 1.97 | 94.69 | 183.46 | 574.77 | 7.49 | 7.53 | 92.92 | 140.39 |
| concurrent | 2 | - | - | 33.06 | 17 | 15 | 0.64 | 1.93 | 3.04 | 89.06 | 276.56 | 543.92 | 9.62 | 9.65 | 165.50 | 221.53 |
| concurrent | 3 | - | - | 32.90 | 16 | 13 | 0.62 | 2.84 | 4.57 | 82.38 | 298.53 | 1630.01 | 10.10 | 10.14 | 174.02 | 224.67 |
| concurrent | 4 | - | - | 32.67 | 20 | 16 | 0.78 | 3.91 | 5.05 | 80.65 | 265.39 | 2829.71 | 10.10 | 10.14 | 185.14 | 246.97 |
| concurrent | 5 | - | - | 33.34 | 18 | 13 | 0.83 | 4.40 | 5.30 | 73.22 | 241.93 | 3018.37 | 9.88 | 9.92 | 156.22 | 216.39 |
| concurrent | 6 | - | - | 33.01 | 18 | 12 | 0.73 | 5.12 | 6.97 | 68.39 | 259.14 | 4731.11 | 10.12 | 10.16 | 148.01 | 197.67 |
| concurrent | 7 | - | - | 33.26 | 21 | 14 | 0.92 | 5.97 | 6.52 | 68.10 | 206.00 | 4762.71 | 10.15 | 10.20 | 143.64 | 205.29 |
| concurrent | 8 | - | - | 32.87 | 21 | 13 | 0.89 | 6.73 | 7.55 | 62.76 | 219.60 | 6454.77 | 10.17 | 10.22 | 139.71 | 195.00 |
| concurrent | 9 | - | - | 32.88 | 21 | 12 | 0.83 | 7.13 | 8.54 | 58.62 | 206.57 | 8087.35 | 10.35 | 10.40 | 114.89 | 163.25 |
| throughput | - | - | - | 32.88 | 21 | 12 | 0.83 | 7.13 | 8.54 | 58.62 | 206.57 | 8087.35 | 10.35 | 10.40 | 114.89 | 163.25 |

### Key Metrics Explained

- **requests_per_second**: Number of requests processed per second
- **request_latency**: Average response time in seconds
- **time_to_first_token_ms**: Time to receive the first token (milliseconds)
- **output_tokens_per_second**: Tokens generated per second
- **tokens_per_second**: Total tokens (input + output) processed per second
- **request_concurrency**: Average number of concurrent requests during the test

## Installation

We use `uv` for the Benchmark. Sync your environment via:

```sh
uv sync
```

### Lightning AI Setup (for automated studio management)

For Lightning AI runners, install and configure the Lightning CLI:

```bash
# Install Lightning CLI
pip install lightning

# Add to your PATH (choose the appropriate one for your setup)
export PATH="$HOME/.local/bin:$PATH"                    # For pip installs
export PATH="$HOME/Library/Python/3.11/bin:$PATH"      # For macOS Python 3.11

# Login to Lightning AI
lightning login
```

**Platform Support:**
- ✅ **macOS/Linux/WSL**: Automatic studio creation and cleanup
- ❌ **Windows**: Manual studio management required (try WSL for automation)

**Note**: On supported platforms, Lightning AI studios are automatically created, started, and stopped. No manual studio management needed!

## Usage

To run the Benchmark, use the following command:

First set the enviroments

### Enviroments
- `MONGODB_URL`: `"mongodb://<username>:<password>@<host>:<port>/<database>?options"` — URL that contains benchmark and API utilities.
- `HF_TOKEN`: `"123123"` — Hugging Face access token, required for accessing private models.
- `RUNPOD_API_KEY`: `"sk_12321313"` — Runpod API key for creating and managing pods.
- `LIGHTNING_USER_ID`: `"123123"` — Lightning AI user ID.
- `LIGHTNING_API_KEY`: `"123123"` — Lightning AI API key.
- `NGROK_AUTH_TOKEN`: `"123123"` — ngrok authentication token for creating public tunnels.
- `SCW_ACCESS_KEY`: `"123123"` — Scaleway access key.
- `SCW_SECRET_KEY`: `"123123"` — Scaleway secret key.
- `SCW_DEFAULT_ORGANIZATION_ID`: `"123123"` — Scaleway default organization ID.
- `SCW_DEFAULT_PROJECT_ID`: `"123123"` — Scaleway default project ID.

After creating .env file with these variables you should run:

### General Usage

```sh
uv run --env-file=.env python <runner>.py \
  --inference_engine <engine> \
  --studio_name "MyStudio" \
  --teamspace "firstbatch/research" \
  --org "firstbatch" \
  --llm_id "<model_id>" \
  --gpu_id "<gpu_type>" \
  --gpu_count 1 \
  --llm_parameter_size "<size>" \
  --llm_common_name "<display_name>"
```

### Examples by Platform

**RunPod:**
```sh
uv run --env-file=.env python runpod_runner.py \
  --inference_engine tensorrt \
  --gpu_id "NVIDIA H200" \
  --volume_in_gb 1000 \
  --container_disk_in_gb 500 \
  --llm_id "Qwen/Qwen3-8B" \
  --llm_parameter_size "8b" \
  --llm_common_name "Qwen3 8B" \
  --gpu_count 1
```

**Modal:**
```sh
uv run --env-file=.env python modal_runner.py \
  --inference_engine vllm \
  --llm_id "Qwen/Qwen3-8B" \
  --gpu_id "A100" \
  --gpu_count 1 \
  --llm_parameter_size "8b" \
  --llm_common_name "Qwen3 8B" \
  --port 8000 \
  --fast_boot true
```

**Lightning AI (Auto-managed on macOS/Linux/WSL):**
```sh
uv run --env-file=.env python lightning_ai_runner.py \
  --inference_engine tensorrt \
  --studio_name "MyStudio" \
  --teamspace "firstbatch/research" \
  --org "firstbatch" \
  --llm_id "Qwen/Qwen3-8B" \
  --gpu_id "L4" \
  --gpu_count 1 \
  --llm_parameter_size "8b" \
  --llm_common_name "Qwen3 8B"
```

**Note on Modal:** Modal automatically manages server lifecycle - servers are created, benchmarks run, and then automatically cleaned up. Cost is calculated based on GPU pricing and runtime. Before running benchmarks, install Modal CLI and authenticate:
```sh
pip install modal
modal setup
```
If `modal setup` doesn't work, try `python -m modal setup`.

### Model ID Formats by Engine

- **Ollama**: `"qwen2:7b"`, `"llama3:8b"`, `"mistral:7b"`
- **vLLM/SGLang/TensorRT**: `"Qwen/Qwen3-8B"`, `"meta-llama/Llama-3-8B"`
- **LMStudio**: HuggingFace model paths

### TensorRT-LLM Optional Parameters

TensorRT-LLM supports several optional parameters that can be passed via CLI to fine-tune performance and memory usage. These parameters are automatically converted into a YAML configuration file that TensorRT-LLM reads at runtime. The key thing to understand is that these are completely optional.

**How it works:** When you provide optional parameters via CLI (like `--moe_backend` or `--ep_size`), the system creates a temporary YAML configuration file inside the container. This YAML file is then passed to TensorRT-LLM's serve command which reads these settings and applies them during model initialization. This approach gives you flexibility to tune performance without needing to manually create configuration files.

**Available optional parameters by model:**

#### Qwen3-30B-A3B (MoE Model)
- `--ep_size` (integer): Expert parallelism size for the MoE layers. Controls how experts are distributed across GPUs.

**Example:**
```sh
uv run --env-file=.env python runpod_runner.py --inference_engine tensorrt \
  --gpu_id "NVIDIA H200" \
  --volume_in_gb 1000 \
  --container_disk_in_gb 500 \
  --llm_id "Qwen/Qwen3-30B-A3B" \
  --llm_parameter_size "30B" \
  --llm_common_name "Qwen3 30B A3B" \
  --gpu_count 1 \
  --port 8000 \
  --ep_size 1
```

#### GPT-OSS-120B (MoE Model)
- `--moe_backend` (string): MoE kernel backend selection. Options: `TRITON` (recommended for H200 GPUs), `CUTLASS` (high throughput), or `TRTLLM` (default). TRITON requires TensorRT-LLM 1.1.0rc1+.
- `--kv_cache_free_gpu_memory_fraction` (float): Fraction of free GPU memory to allocate for KV cache. Values between 0.0 and 1.0. Higher values allow more concurrent requests but may cause OOM errors.
- `--enable_attention_dp` (string): Enable attention data parallelism. Use `"true"` for maximum throughput scenarios, `"false"` for low-latency use cases. Default is `false`.
- `--ep_size` (integer): Expert parallelism size for distributing MoE experts across GPUs.

**Example:**
```sh
uv run --env-file=.env python runpod_runner.py --inference_engine tensorrt \
  --gpu_id "NVIDIA H200" \
  --volume_in_gb 1000 \
  --container_disk_in_gb 500 \
  --llm_id "openai/gpt-oss-120B" \
  --llm_parameter_size "120b" \
  --llm_common_name "GPT-OSS 120B" \
  --gpu_count 1 \
  --port 8000 \
  --moe_backend TRITON \
  --kv_cache_free_gpu_memory_fraction 0.9 \
  --enable_attention_dp "false" \
  --ep_size 1
```

#### Kimi-K2-Instruct (MoE Model)
- `--moe_backend` (string, optional): MoE kernel backend selection. **Only `TRTLLM` is supported** for Kimi-K2 (added in TensorRT-LLM 1.2.0rc2, [PR #7761](https://github.com/NVIDIA/TensorRT-LLM/pull/7761)). If not provided, uses PyTorch backend by default.
- `--ep_size` (integer, optional): Expert parallelism size for distributing MoE experts across GPUs.
- `--enable_attention_dp` (string, optional): Enable attention data parallelism. Use `"true"` for maximum throughput scenarios, `"false"` for low-latency use cases. Default is `false`.

**Important Notes:**
- Kimi-K2 is a very large model (1 trillion total parameters, 32B activated) requiring substantial GPU resources.
- Single GPU setups will not work due to memory constraints.

**Example:**
```sh
uv run --env-file=.env python runpod_runner.py --inference_engine tensorrt \
  --gpu_id "NVIDIA H200" \
  --volume_in_gb 1500 \
  --container_disk_in_gb 1500 \
  --llm_id "moonshotai/Kimi-K2-Instruct" \
  --llm_parameter_size "32B" \
  --llm_common_name "Kimi-K2" \
  --gpu_count 8 \
  --port 8000 \
  --ep_size 8
```

#### DeepSeek-R1/V3
- `--ep_size` (integer, optional): Expert parallelism size for MoE layers. Controls how experts are distributed across GPUs.
- `--pp_size` (integer, optional): Pipeline parallelism size for multi-GPU setups.
- `--kv_cache_free_gpu_memory_fraction` (float, optional): Fraction of free GPU memory to allocate for KV cache. Values between 0.0 and 1.0.
- `--enable_attention_dp` (string, optional): Enable attention data parallelism. Use `"true"` for maximum throughput, `"false"` for low-latency. Default is `false`.

**Example:**
```sh
uv run --env-file=.env python runpod_runner.py --inference_engine tensorrt \
  --gpu_id "NVIDIA H200" \
  --volume_in_gb 1000 \
  --container_disk_in_gb 500 \
  --llm_id "deepseek-ai/DeepSeek-R1" \
  --llm_parameter_size "7b" \
  --llm_common_name "DeepSeek R1" \
  --gpu_count 1 \
  --port 8000 \
  --ep_size 1 \
  --pp_size 1
```




## 🔧 Supported Configurations

Will be added soon <>

## 🔗 Related Links

- **Web Platform**: [dria.co/inference-benchmark](https://dria.co/inference-benchmark)
- **Dria Main Site**: [dria.co](https://dria.co)
- **Documentation**: Available at `/docs` when running the API server


