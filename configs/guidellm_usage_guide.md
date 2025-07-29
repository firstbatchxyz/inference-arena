# GuideLLM Benchmark Client Usage Guide

This custom benchmark client extends GuideLLM to support:
1. Custom endpoint paths (e.g., `/api/generate` instead of `/v1/completions`)
2. MongoDB integration for loading prompts
3. Easy configuration for various benchmarking scenarios

## Installation

Make sure you have the required dependencies:
```bash
pip install guidellm pymongo httpx loguru datasets transformers
```

## Basic Usage

### 1. Simple Benchmark with Custom Endpoint

```python
from guidellm_benchmark_client import GuideLLMBenchmarkClient

# Create client with custom endpoint
client = GuideLLMBenchmarkClient(
    base_url="http://localhost:8000",
    model="llama2",
    text_completions_path="/api/generate",  # Your custom endpoint
    max_output_tokens=100,
)

# Run benchmark with custom prompts
prompts = [
    {"prompt": "What is machine learning?"},
    {"prompt": "Explain quantum computing"},
    {"prompt": "How does blockchain work?"},
]

report, saved_path = client.run_sync(
    data=prompts,
    rate_type="synchronous",
    max_requests=10,
    output_path="benchmark_results.json",
)
```

### 2. MongoDB Integration

```python
# Create client with MongoDB connection
client = GuideLLMBenchmarkClient(
    base_url="http://localhost:8000",
    model="llama2",
    mongo_url="mongodb://localhost:27017/",
    text_completions_path="/api/generate",
    max_output_tokens=100,
)

# Run benchmark with prompts from MongoDB
report, path = client.run_sync(
    mongo_query={"category": "science"},  # Filter prompts
    mongo_limit=100,  # Limit number of prompts
    rate_type="synchronous",
    max_requests=50,
    output_path="mongodb_benchmark_results.json",
)
```

### 3. Advanced Benchmarking Options

```python
# Concurrent requests benchmark
report, path = client.run_sync(
    data=prompts,
    rate_type="concurrent",
    rate=10,  # 10 concurrent requests
    max_seconds=60,  # Run for 60 seconds
    warmup_percent=0.1,  # 10% warmup
    show_progress=True,
)

# Throughput benchmark (requests per second)
report, path = client.run_sync(
    data=prompts,
    rate_type="throughput",
    max_requests=100,
    output_path="throughput_benchmark.json",
)

# Sweep benchmark (multiple rates)
report, path = client.run_sync(
    data=prompts,
    rate_type="sweep",
    rate=[1, 5, 10, 20],  # Test at different rates
    max_seconds=30,  # 30 seconds per rate
    output_path="sweep_benchmark.json",
)
```

## Available Rate Types

- **synchronous**: One request at a time
- **throughput**: Maximum throughput test
- **concurrent**: Fixed number of concurrent requests
- **async**: Fixed requests per second
- **constant**: Constant rate (requests/sec)
- **poisson**: Poisson distribution rate
- **sweep**: Test multiple rates

## MongoDB Schema

The client expects MongoDB documents with one of these fields:
- `prompt`: The prompt text
- `text`: Alternative field for prompt
- `content`: Another alternative field

Example MongoDB document:
```json
{
    "_id": "...",
    "prompt": "What is artificial intelligence?",
    "category": "science",
    "difficulty": "medium"
}
```

## Custom Backend Configuration

The client supports additional backend configurations:

```python
client = GuideLLMBenchmarkClient(
    base_url="http://localhost:8000",
    model="llama2",
    text_completions_path="/api/generate",
    chat_completions_path="/api/chat",  # For chat completions
    api_key="your-api-key",  # If authentication is needed
    max_output_tokens=200,
)
```

## Output Analysis

The benchmark results are saved in JSON format and include:
- Request/response times
- Token counts
- Throughput metrics
- Latency statistics
- Error rates

You can load and analyze the results:

```python
import json

with open("benchmark_results.json", "r") as f:
    results = json.load(f)

# Access benchmark statistics
for benchmark in results["benchmarks"]:
    print(f"Rate: {benchmark['rate']}")
    print(f"Mean latency: {benchmark['metrics']['mean_latency']}")
    print(f"Throughput: {benchmark['metrics']['throughput']}")
```

## Troubleshooting

1. **Connection errors**: Ensure your server is running and accessible
2. **MongoDB connection**: Check MongoDB URL and credentials
3. **Custom endpoints**: Verify the endpoint path matches your server API
4. **Model not found**: Ensure the model name matches what's available on the server

## Example Script

Here's a complete example script:

```python
#!/usr/bin/env python3
import asyncio
from guidellm_benchmark_client import GuideLLMBenchmarkClient

async def main():
    # Initialize client
    client = GuideLLMBenchmarkClient(
        base_url="http://localhost:8000",
        model="llama2",
        mongo_url="mongodb://localhost:27017/",
        text_completions_path="/api/generate",
        max_output_tokens=100,
    )
    
    # Run different benchmark types
    print("Running synchronous benchmark...")
    report1, _ = await client.run_benchmark(
        mongo_limit=50,
        rate_type="synchronous",
        max_requests=10,
        output_path="sync_benchmark.json",
    )
    
    print("Running concurrent benchmark...")
    report2, _ = await client.run_benchmark(
        mongo_limit=100,
        rate_type="concurrent",
        rate=5,
        max_seconds=30,
        output_path="concurrent_benchmark.json",
    )
    
    print("Benchmarks completed!")

if __name__ == "__main__":
    asyncio.run(main())
``` 