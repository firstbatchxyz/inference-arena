# Dria Benchmark System

A comprehensive benchmarking platform for comparing LLM (Large Language Model) performance across different GPU configurations and inference engines. Currently supports RunPod as the cloud infrastructure provider.

## 🎯 Project Aim

We aim to show and allow users to compare performances of different LLM models across various GPU configurations and inference engines. The system currently supports:

- **Cloud Provider**: RunPod
- **Inference Engines**: Ollama, SGLang, vLLM
- **Deployment**: Same LLM models with different inference engines on the same GPU for fair performance comparison

## 🌐 View Benchmark Results

**📊 [Visit dria.co/inference-benchmark](https://dria.co/inference-benchmark) to view and compare benchmark results in real-time!**

The web platform provides:
- Interactive benchmark comparisons
- Real-time GPU pricing
- Performance analytics and insights
- Community discussions and comments
- AI-powered recommendations

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   MongoDB        │    │   RunPod        │
│   Backend       │◄──►│   Database       │    │   GPU Cloud     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Assistant  │    │   Price Scraper  │    │   Benchmark     │
│   (OpenAI)      │    │   (Multi-Platform)│    │   Runners       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## ✨ Key Features

- **Multi-Engine Benchmarking**: Compare Ollama, vLLM, and SGLang performance
- **Cloud GPU Integration**: Automated RunPod provisioning and management
- **Real-time GPU Pricing**: Track pricing across multiple cloud providers
- **AI-Powered Analysis**: Chat assistant for benchmark insights and recommendations
- **Community Platform**: Comments, upvotes, and collaborative discussions
- **Cost Optimization**: Intelligent GPU selection and cost estimation

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

For each benchmark, we create **10 different benchmarks**:
1. **1 Throughput benchmark** (maximum capacity test)
2. **9 Concurrent benchmarks** (rates 1-9)

All 10 benchmarks are recorded and displayed to users for comprehensive performance analysis.

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

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MongoDB
- RunPod account and API key
- HuggingFace token (for model access)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dkn-benchmark-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export MONGODB_URL="your_mongodb_connection_string"
   export RUNPOD_API_KEY="your_runpod_api_key"
   export HF_TOKEN="your_huggingface_token"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

4. Run a benchmark:
   ```bash
   # Ollama benchmark
   python benchmark/run_ollama_benchmark_with_runpod.py \
     --gpu_id "NVIDIA H200" \
     --volume_in_gb 100 \
     --container_disk_in_gb 50 \
     --llm_id "llama2:7b" \
     --llm_parameter_size "7b" \
     --llm_common_name "Llama2-7B"
   
   # vLLM benchmark
   python benchmark/run_vllm_benchmark_with_runpod.py \
     --gpu_id "NVIDIA H200" \
     --volume_in_gb 100 \
     --container_disk_in_gb 50 \
     --llm_id "meta-llama/Llama-2-7b-chat-hf" \
     --llm_parameter_size "7b" \
     --llm_common_name "Llama2-7B"
   
   # SGLang benchmark
   python benchmark/run_sglang_benchmark_with_runpod.py \
     --gpu_id "NVIDIA H200" \
     --volume_in_gb 100 \
     --container_disk_in_gb 50 \
     --llm_id "meta-llama/Llama-2-7b-chat-hf" \
     --llm_parameter_size "7b" \
     --llm_common_name "Llama2-7B"
   ```

## 🔧 Supported Configurations

### Models (42 Available)
| Model Family | Variants | Parameter Sizes |
|-------------|----------|-----------------|
| **Llama 3.1** | Instruct | 70B, 405B |
| **Qwen 3** | Base | 14B, 32B, 235B (MoE) |
| **Gemma 3** | Base | 1B, 4B, 12B |
| **Mistral** | Small, Devstral | 7.3B, 24B |
| **Falcon** | Base | 10B, 180B |
| **DeepSeek R1** | MoE | 671B (~37B active) |

### GPU Hardware (42 Supported)
| Category | Examples | Memory Range |
|----------|----------|--------------|
| **Data Center** | H100, H200, A100, B200 | 40GB - 192GB |
| **Consumer** | RTX 5090, 4090, 3090 | 12GB - 24GB |
| **Workstation** | RTX 6000 Ada, A6000 | 24GB - 48GB |

### Inference Engines
| Engine | Docker Image | API Endpoint | Port | Special Features |
|--------|--------------|--------------|------|------------------|
| **Ollama** | `ollama/ollama:latest` | `/api/generate` | 11434 | Flash attention, GPU overhead control |
| **vLLM** | `vllm/vllm-openai:v0.6.0` | `/v1/completions` | 8000 | CUDA optimization, tensor parallelism |
| **SGLang** | `lmsysorg/sglang:latest` | `/v1/completions` | 30000 | Multi-GPU support, KV cache optimization |

## 📊 API Usage

### Start the API Server
```bash
cd api
uvicorn main:app --reload
```

### Get Benchmark Results
```bash
# Get specific benchmark results
curl "http://localhost:8000/benchmark_results?platform=Ollama&gpu_type=NVIDIA%20H200&llm_id=llama2:7b"

# Get all benchmarks
curl "http://localhost:8000/get_all_benchmarks"

# Filter results
curl "http://localhost:8000/get_benchmark_results_by_filters?model_name=llama2&inference_engine=Ollama"
```

### AI Analysis
```bash
curl -X POST "http://localhost:8000/get_response_from_ai" \
  -H "Content-Type: application/json" \
  -d '{"message": "Which inference engine performs best for 7B models?", "pod_id": "your_pod_id"}'
```

## 🗄️ Database Schema

### Collections

1. **`pod_benchmarks`**: Pod creation and configuration details
2. **`benchmark_results`**: Detailed benchmark results for each test
3. **`benchmark_suite_results`**: Summary of benchmark suites
4. **`users`**: User information and authentication data
5. **`comments`**: User comments on benchmark results

### Key Fields
- `pod_id`: Unique identifier for each benchmark run
- `benchmark_type`: "concurrent" or "throughput"
- `rate`: Concurrency level (1-9 for concurrent tests)
- `requests_per_second`: Performance metric
- `output_tokens_per_second`: Token generation speed
- `time_to_first_token_ms`: Latency metric

## 🔍 Performance Analysis

The system provides comprehensive performance analysis including:

- **Throughput Comparison**: RPS and TPS across different engines
- **Latency Analysis**: Response times and time-to-first-token
- **Resource Utilization**: GPU memory and compute efficiency
- **Cost Analysis**: Per-hour costs and cost-effectiveness
- **Scalability**: Performance at different concurrency levels

## 🌐 Web Platform Features

**Visit [dria.co/inference-benchmark](https://dria.co/inference-benchmark) for:**

- **Interactive Benchmark Comparisons**: Side-by-side performance analysis
- **Real-time GPU Pricing**: Up-to-date cost information from major providers
- **Advanced Filtering**: Filter by model, engine, GPU, performance metrics
- **Community Features**: Comments, likes, and discussions on benchmark results
- **AI-Powered Insights**: Get recommendations and analysis from AI
- **Export Capabilities**: Download benchmark data for further analysis
- **Performance Trends**: Historical performance tracking and trends

## AI Assistant

The integrated AI assistant provides:
- **Benchmark Analysis** for performance comparisons and insights
- **Cost Analysis** for GPU and configuration recommendations
- **Trend Analysis** for performance patterns

Sample interactions:
- *"Which inference engine is fastest for 70B models on H100?"*
- *"What's the most cost-effective GPU for Qwen-32B?"*
- *"Show me latency trends for vLLM benchmarks this month"*

## GPU Selection
The system automatically selects optimal GPU configurations based on:
- Model parameter size and memory requirements
- Cost efficiency analysis
- Historical performance data
- Current GPU availability and pricing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a Pull Request


## 🔗 Related Links

- **Web Platform**: [dria.co/inference-benchmark](https://dria.co/inference-benchmark)
- **Dria Main Site**: [dria.co](https://dria.co)
- **Documentation**: Available at `/docs` when running the API server


