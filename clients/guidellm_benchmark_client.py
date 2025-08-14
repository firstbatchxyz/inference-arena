import asyncio
import random
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from datasets import Dataset
from guidellm.backend import Backend
from guidellm.backend.openai import (
    OpenAIHTTPBackend,
    ResponseSummary,
    StreamingTextResponse,
)
from guidellm.benchmark import benchmark_generative_text
from guidellm.dataset import InMemoryDatasetCreator

# Import local mongo client
from clients.mongo_client import Mongo

# Import tokenizer mappings
from utils.ollama_utils import OLLAMA_TO_HF_TOKENIZER


class CustomOpenAIBackend(OpenAIHTTPBackend):
    """Extended OpenAI backend that supports custom endpoint paths"""

    def __init__(
        self,
        target: str | None = None,
        text_completions_path: str | None = None,
        chat_completions_path: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: float | None = None,
        http2: bool | None = True,
        max_output_tokens: int | None = None,
    ):
        super().__init__(
            target=target,
            model=model,
            api_key=api_key,
            organization=organization,
            project=project,
            timeout=timeout,
            http2=http2,
            max_output_tokens=max_output_tokens,
        )

        # Override default paths if provided
        self._text_completions_path = text_completions_path or "/v1/completions"
        self._chat_completions_path = chat_completions_path or "/v1/chat/completions"

    def _extract_ollama_delta_content(self, data: dict) -> str | None:
        """Extract streaming delta content from Ollama response format."""
        # Ollama uses "response" field for streaming content
        return data.get("response", "")

    async def _iterative_completions_request(
        self,
        type_: Literal["text_completions", "chat_completions"],
        request_id: str | None,
        request_prompt_tokens: int | None,
        request_output_tokens: int | None,
        headers: dict,
        params: dict,
        payload: dict,
    ) -> AsyncGenerator[StreamingTextResponse | ResponseSummary, None]:
        # Import necessary modules for the implementation
        import json
        import time

        from guidellm.backend.response import RequestArgs

        # Use custom paths
        if type_ == "text_completions":
            target = f"{self.target}{self._text_completions_path}"
        elif type_ == "chat_completions":
            target = f"{self.target}{self._chat_completions_path}"
        else:
            raise ValueError(f"Unsupported type: {type_}")

        # Ensure streaming is enabled in the payload
        payload = payload.copy()  # Don't modify the original
        payload["stream"] = True

        # Remove unsupported parameters for vLLM compatibility
        unsupported_params = ["stream_options", "max_completion_tokens", "ignore_eos"]
        for param in unsupported_params:
            payload.pop(param, None)

        # For Ollama compatibility, ensure we have required fields
        if type_ == "text_completions" and "prompt" not in payload:
            # Convert messages to prompt for text completions
            if "messages" in payload:
                payload["prompt"] = payload.pop("messages")[-1].get("content", "")

        response_value = ""
        response_prompt_count: int | None = None
        response_output_count: int | None = None
        iter_count = 0
        start_time = time.time()
        iter_time = start_time
        first_iter_time: float | None = None
        last_iter_time: float | None = None

        yield StreamingTextResponse(
            type_="start",
            value="",
            start_time=start_time,
            first_iter_time=None,
            iter_count=iter_count,
            delta="",
            time=start_time,
            request_id=request_id,
        )

        # Reset start time after yielding start response
        start_time = time.time()

        async with self._get_async_client().stream(
            "POST", target, headers=headers, params=params, json=payload
        ) as stream:
            stream.raise_for_status()

            async for line in stream.aiter_lines():
                iter_time = time.time()

                if not line or not line.strip():
                    continue

                # Handle both OpenAI-style and Ollama-style streaming
                try:
                    if line.strip().startswith("data:"):
                        # OpenAI-style streaming
                        if line.strip() == "data: [DONE]":
                            break
                        data = json.loads(line.strip()[len("data: ") :])
                    else:
                        # Ollama-style streaming (direct JSON)
                        data = json.loads(line.strip())

                        # Check if this is the final response from Ollama
                        if data.get("done", False):
                            # Extract usage information from final response
                            if "prompt_tokens" in data:
                                response_prompt_count = data["prompt_tokens"]
                            if "completion_tokens" in data:
                                response_output_count = data["completion_tokens"]
                            # Process any remaining content and break
                            if delta := self._extract_ollama_delta_content(data):
                                if first_iter_time is None:
                                    first_iter_time = iter_time
                                last_iter_time = iter_time
                                iter_count += 1
                                response_value += delta

                                yield StreamingTextResponse(
                                    type_="iter",
                                    value=response_value,
                                    iter_count=iter_count,
                                    start_time=start_time,
                                    first_iter_time=first_iter_time,
                                    delta=delta,
                                    time=iter_time,
                                    request_id=request_id,
                                )
                            break

                    # Extract delta content
                    delta = None
                    if line.strip().startswith("data:"):
                        # OpenAI format
                        delta = self._extract_completions_delta_content(type_, data)
                    else:
                        # Ollama format
                        delta = self._extract_ollama_delta_content(data)

                    if delta:
                        if first_iter_time is None:
                            first_iter_time = iter_time
                        last_iter_time = iter_time

                        iter_count += 1
                        response_value += delta

                        yield StreamingTextResponse(
                            type_="iter",
                            value=response_value,
                            iter_count=iter_count,
                            start_time=start_time,
                            first_iter_time=first_iter_time,
                            delta=delta,
                            time=iter_time,
                            request_id=request_id,
                        )

                    # Extract usage information (OpenAI format)
                    if usage := self._extract_completions_usage(data):
                        response_prompt_count = usage["prompt"]
                        response_output_count = usage["output"]

                except json.JSONDecodeError:

                    continue

        yield ResponseSummary(
            value=response_value,
            request_args=RequestArgs(
                target=target,
                headers=headers,
                params=params,
                payload=payload,
                timeout=self.timeout,
                http2=self.http2,
            ),
            start_time=start_time,
            end_time=iter_time,
            first_iter_time=first_iter_time,
            last_iter_time=last_iter_time,
            iterations=iter_count,
            request_prompt_tokens=request_prompt_tokens,
            request_output_tokens=request_output_tokens,
            response_prompt_tokens=response_prompt_count,
            response_output_tokens=response_output_count,
            request_id=request_id,
        )


# Register the custom backend
Backend._registry["custom_openai"] = CustomOpenAIBackend


class MongoDatasetLoader:
    """Load prompts from MongoDB and convert to guidellm dataset"""

    def __init__(
        self,
        mongo_url: str,
        db_name: str = "dria_benchmark",
        collection_name: str = "benchmark_test_data",
    ):
        self.mongo = Mongo(mongo_url)
        self.db_name = db_name
        self.collection_name = collection_name

    def load_prompts(
        self, query: dict | None = None, limit: int | None = None
    ) -> list[dict]:
        """Load prompts from MongoDB"""
        query = query or {}
        cursor = self.mongo.find_many(self.collection_name, query)

        if limit:
            cursor = cursor.limit(limit)

        prompts = []
        for doc in cursor:
            # Handle different prompt formats
            if "prompt" in doc:
                prompts.append({"prompt": doc["prompt"]})

        return prompts

    def create_dataset(
        self, query: dict | None = None, limit: int | None = None
    ) -> Dataset:
        """Create a Hugging Face Dataset from MongoDB prompts"""
        prompts = self.load_prompts(query, limit)
        if not prompts:
            raise ValueError("No prompts found in MongoDB")

        # Create dataset using InMemoryDatasetCreator
        dataset = InMemoryDatasetCreator.handle_create(
            data=prompts,
            data_args=None,
            processor=None,
            processor_args=None,
            random_seed=42,
        )

        return dataset


class GuideLLMBenchmarkClient:
    """Main benchmark client that integrates guidellm with custom configuration
    This client includes:
    - Custom OpenAI backend support for various endpoints
    - MongoDB integration for loading prompts
    - Exponential backoff retry mechanisms
    - Connection timeout handling
    - Graceful failure recovery
    - Detailed logging and monitoring
    - Health check capabilities
    - Benchmark suite execution
    """

    # Add tokenizer mapping as class attribute
    OLLAMA_TO_HF_TOKENIZER = OLLAMA_TO_HF_TOKENIZER

    def __init__(
        self,
        base_url: str,
        model: str,
        mongo_url: str | None = None,
        text_completions_path: str | None = None,
        chat_completions_path: str | None = None,
        api_key: str | None = None,
        max_output_tokens: int | None = 100,
        processor: str | None = None,  # Allow override of processor/tokenizer
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 300.0,
        health_check_timeout: float = 30.0,
    ):
        """
        Initialize benchmark client with custom configuration and retry mechanisms.
        Args:
            base_url: Base URL of the model server
            model: Model name/identifier
            mongo_url: MongoDB connection URL
            text_completions_path: Path for text completions endpoint
            chat_completions_path: Path for chat completions endpoint
            api_key: Optional API key
            max_output_tokens: Maximum output tokens
            processor: Custom processor/tokenizer
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            timeout: Request timeout (seconds)
            health_check_timeout: Health check timeout (seconds)
        """
        self.base_url = base_url
        self.model = model
        self.mongo_url = mongo_url
        self.text_completions_path = text_completions_path or "/api/generate"
        self.chat_completions_path = chat_completions_path or "/v1/chat/completions"
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.processor = processor  # Store custom processor

        # Retry and timeout configuration
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.health_check_timeout = health_check_timeout

        # Initialize MongoDB loader if URL provided
        self.mongo_loader = MongoDatasetLoader(mongo_url) if mongo_url else None
        self.mongo_client = Mongo(mongo_url) if mongo_url else None

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)

    @contextmanager
    def _timeout_context(self, timeout: float):
        """Context manager for operation timeouts."""

        def timeout_handler():
            pass

        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check on the model server.

        Returns:
            Dict with health status, response time, and server info
        """
        health_status = {
            "healthy": False,
            "response_time_ms": None,
            "server_info": None,
            "error": None,
            "timestamp": time.time(),
        }

        try:
            with self._timeout_context(self.health_check_timeout):
                start_time = time.time()

                # Try to get model list
                response = httpx.get(
                    f"{self.base_url}/v1/models", timeout=self.health_check_timeout
                )

                response_time = (time.time() - start_time) * 1000
                health_status["response_time_ms"] = response_time

                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        model_found = any(
                            model.get("id") == self.model for model in data["data"]
                        )

                        if model_found:
                            health_status["healthy"] = True
                            health_status["server_info"] = {
                                "available_models": [m.get("id") for m in data["data"]],
                                "target_model": self.model,
                                "response_code": response.status_code,
                            }
                        else:
                            health_status["error"] = (
                                f"Model {self.model} not found in available models"
                            )
                    else:
                        health_status["error"] = "No models available on server"
                else:
                    health_status["error"] = (
                        f"Server returned status {response.status_code}"
                    )

        except httpx.TimeoutException:
            health_status["error"] = (
                f"Health check timed out after {self.health_check_timeout}s"
            )
        except httpx.ConnectError:
            health_status["error"] = "Cannot connect to server"
        except Exception as e:
            health_status["error"] = f"Health check failed: {str(e)}"

        return health_status

    def wait_for_server_ready(
        self, max_wait_time: float = 600.0, check_interval: float = 10.0
    ) -> bool:
        """
        Wait for server to be ready with exponential backoff.

        Args:
            max_wait_time: Maximum time to wait (seconds)
            check_interval: Initial interval between checks (seconds)

        Returns:
            True if server becomes ready, False if timeout
        """
        start_time = time.time()
        attempt = 0

        while time.time() - start_time < max_wait_time:
            health = self.health_check()

            if health["healthy"]:
                return True

            # Calculate next check interval with backoff
            wait_time = min(check_interval * (1.5**attempt), 60.0)

            time.sleep(wait_time)
            attempt += 1

        return False

    async def run_benchmark(
        self,
        data: str | Path | list | None = None,
        mongo_query: dict | None = None,
        mongo_limit: int | None = None,
        rate_type: str = "synchronous",
        rate: int | float | list | None = None,
        max_seconds: float | None = None,
        max_requests: int | None = None,
        warmup_percent: float | None = 0.1,
        output_path: str | Path | None = None,
        show_progress: bool = True,
        disable_token_counting: bool = False,
    ):
        """Run benchmark with custom configuration

        Args:
            disable_token_counting: If True, disables token counting (useful for non-HF models)
        """

        # Prepare data
        if self.mongo_loader and (mongo_query is not None or data is None):
            # Load from MongoDB
            dataset = self.mongo_loader.create_dataset(mongo_query, mongo_limit)
            data_source = dataset
        else:
            # Use provided data
            data_source = data or []

        # Create backend arguments
        backend_args = {
            "text_completions_path": self.text_completions_path,
            "chat_completions_path": self.chat_completions_path,
            "api_key": self.api_key,
            "max_output_tokens": self.max_output_tokens,
        }

        # Determine processor/tokenizer
        if disable_token_counting:
            processor = None
        elif self.processor:
            # Use custom processor if provided
            processor = self.processor
        elif self.model in self.OLLAMA_TO_HF_TOKENIZER:
            # Map Ollama model to HF tokenizer
            processor = self.OLLAMA_TO_HF_TOKENIZER[self.model]
        else:
            # Default to None to avoid errors with unknown models
            processor = None

        # Test tokenizer loading if one is specified
        if processor and not disable_token_counting:
            try:
                from guidellm.utils.hf_transformers import check_load_processor

                # Test load the processor to ensure it works
                check_load_processor(
                    processor,
                    processor_args=None,
                    error_msg="Testing tokenizer availability",
                )
            except Exception:
                processor = None

        # Run benchmark
        report, saved_path = await benchmark_generative_text(
            target=self.base_url,
            backend_type="custom_openai",
            backend_args=backend_args,
            model=self.model,
            processor=processor,
            processor_args=None,
            data=data_source,
            data_args=None,
            data_sampler=None,
            rate_type=rate_type,
            rate=rate,
            max_seconds=max_seconds,
            max_requests=max_requests,
            warmup_percent=warmup_percent,
            cooldown_percent=0.0,
            show_progress=show_progress,
            show_progress_scheduler_stats=False,
            output_console=True,
            output_path=output_path,
            output_extras=None,
            output_sampling=None,
            random_seed=42,
        )

        return report, saved_path

    def run_sync(self, **kwargs):
        """Synchronous wrapper for run_benchmark"""
        return asyncio.run(self.run_benchmark(**kwargs))

    def run_sync_with_retry(
        self,
        max_seconds: int | None = None,
        mongo_query: dict | None = None,
        rate_type: str = "concurrent",
        output_path: str = "benchmark_results.json",
        rate: int | None = None,
        **kwargs,
    ) -> tuple:
        """
        Run benchmark with retry mechanism and error handling.

        Returns:
            Tuple of (report, path) or (None, None) if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:

                # Health check before running benchmark
                if not self.health_check()["healthy"]:
                    raise RuntimeError("Server health check failed before benchmark")

                # Run the actual benchmark
                result = self.run_sync(
                    max_seconds=max_seconds,
                    mongo_query=mongo_query or {},
                    rate_type=rate_type,
                    output_path=output_path,
                    rate=rate,
                    **kwargs,
                )

                return result

            except Exception as e:
                last_exception = e

                # Log the failure to database
                self._log_benchmark_failure(attempt + 1, str(e))

                # Don't retry on the last attempt
                if attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    time.sleep(delay)

        # Log final failure
        self._log_benchmark_final_failure(str(last_exception))
        return None, None

    def run_benchmark_suite(
        self,
        pod_id: str,
        concurrent_rates: list[int] = None,
        include_throughput: bool = True,
        max_seconds_per_test: int = 30,
    ) -> dict[str, Any]:
        """
        Run a complete benchmark suite with error handling.

        Args:
            pod_id: Pod identifier for logging
            concurrent_rates: List of concurrent rates to test
            include_throughput: Whether to include throughput test
            max_seconds_per_test: Maximum seconds per individual test

        Returns:
            Dict with benchmark results and statistics
        """
        if concurrent_rates is None:
            concurrent_rates = list(range(1, 10))  # Default: 1-9 concurrent requests

        suite_results = {
            "pod_id": pod_id,
            "start_time": time.time(),
            "concurrent_results": [],
            "throughput_result": None,
            "failures": [],
            "total_tests": len(concurrent_rates) + (1 if include_throughput else 0),
            "successful_tests": 0,
            "error_summary": {},
        }

        # Test concurrent rates
        for rate in concurrent_rates:

            try:
                report, path = self.run_sync_with_retry(
                    max_seconds=max_seconds_per_test,
                    rate_type="concurrent",
                    rate=rate,
                    output_path=f"benchmark_results_concurrent_{rate}.json",
                )

                if report and report.benchmarks:
                    benchmark_report = report.benchmarks[0]

                    result_data = self._extract_benchmark_metrics(
                        benchmark_report, pod_id, rate
                    )
                    suite_results["concurrent_results"].append(result_data)
                    suite_results["successful_tests"] += 1

                    # Log to database
                    if self.mongo_client:
                        self.mongo_client.insert_one("benchmark_results", result_data)

                else:
                    suite_results["failures"].append(
                        {
                            "test_type": "concurrent",
                            "rate": rate,
                            "error": "No benchmark results returned",
                        }
                    )

            except Exception as e:
                error_msg = str(e)
                suite_results["failures"].append(
                    {"test_type": "concurrent", "rate": rate, "error": error_msg}
                )
                self._track_error_type(suite_results["error_summary"], error_msg)

            # Brief pause between tests
            time.sleep(2)

        # Test throughput
        if include_throughput:

            try:
                report, path = self.run_sync_with_retry(
                    max_seconds=max_seconds_per_test,
                    rate_type="throughput",
                    output_path="benchmark_results_throughput.json",
                )

                if report and report.benchmarks:
                    benchmark_report = report.benchmarks[0]

                    result_data = self._extract_benchmark_metrics(
                        benchmark_report, pod_id, None, "throughput"
                    )
                    suite_results["throughput_result"] = result_data
                    suite_results["successful_tests"] += 1

                    # Log to database
                    if self.mongo_client:
                        self.mongo_client.insert_one("benchmark_results", result_data)

                else:
                    suite_results["failures"].append(
                        {
                            "test_type": "throughput",
                            "error": "No benchmark results returned",
                        }
                    )

            except Exception as e:
                error_msg = str(e)
                suite_results["failures"].append(
                    {"test_type": "throughput", "error": error_msg}
                )
                self._track_error_type(suite_results["error_summary"], error_msg)

        suite_results["end_time"] = time.time()
        suite_results["total_duration"] = (
            suite_results["end_time"] - suite_results["start_time"]
        )
        suite_results["success_rate"] = (
            suite_results["successful_tests"] / suite_results["total_tests"]
        )

        return suite_results

    def _extract_benchmark_metrics(
        self, benchmark_report, pod_id: str, rate: int | None, test_type: str = None
    ) -> dict:
        """Extract metrics from benchmark report."""
        result = {
            "pod_id": pod_id,
            "benchmark_type": test_type or benchmark_report.args.profile.type_,
            "rate": rate,
            "max_number": benchmark_report.args.max_number,
            "warmup_number": benchmark_report.args.warmup_number,
            "benchmark_duration": None,
            "total_requests": benchmark_report.run_stats.requests_made.total,
            "successful_requests": benchmark_report.run_stats.requests_made.successful,
            "requests_per_second": benchmark_report.metrics.requests_per_second.total.mean,
            "request_concurrency": benchmark_report.metrics.request_concurrency.total.mean,
            "request_latency": benchmark_report.metrics.request_latency.total.mean,
            "prompt_token_count": benchmark_report.metrics.prompt_token_count.total.mean,
            "output_token_count": benchmark_report.metrics.output_token_count.total.mean,
            "time_to_first_token_ms": benchmark_report.metrics.time_to_first_token_ms.total.mean,
            "time_per_output_token_ms": benchmark_report.metrics.time_per_output_token_ms.total.mean,
            "inter_token_latency_ms": benchmark_report.metrics.inter_token_latency_ms.total.mean,
            "output_tokens_per_second": benchmark_report.metrics.output_tokens_per_second.total.mean,
            "tokens_per_second": benchmark_report.metrics.tokens_per_second.total.mean,
            "timestamp": time.time(),
        }

        # Calculate benchmark duration if available
        if benchmark_report.run_stats.end_time and benchmark_report.run_stats.start_time:
            result["benchmark_duration"] = (
                benchmark_report.run_stats.end_time - benchmark_report.run_stats.start_time
            )

        # Handle rate extraction for profile types
        if not rate and self.mongo_client:
            result["rate"] = self.mongo_client.safe_get_metric(
                benchmark_report.args.profile, "streams[0]"
            )

        return result

    def _log_benchmark_failure(self, attempt: int, error: str):
        """Log individual benchmark failure to database."""
        if not self.mongo_client:
            return

        try:
            self.mongo_client.insert_one(
                "benchmark_failures",
                {
                    "base_url": self.base_url,
                    "model": self.model,
                    "attempt": attempt,
                    "error": error,
                    "timestamp": time.time(),
                },
            )
        except Exception:
            pass

    def _log_benchmark_final_failure(self, error: str):
        """Log final benchmark failure after all retries."""
        if not self.mongo_client:
            return

        try:
            self.mongo_client.insert_one(
                "benchmark_final_failures",
                {
                    "base_url": self.base_url,
                    "model": self.model,
                    "final_error": error,
                    "max_retries": self.max_retries,
                    "timestamp": time.time(),
                },
            )
        except Exception:
            pass

    def _track_error_type(self, error_summary: dict, error_msg: str):
        """Track error types for analysis."""
        error_type = "unknown"

        if "timeout" in error_msg.lower():
            error_type = "timeout"
        elif "connection" in error_msg.lower():
            error_type = "connection"
        elif "server" in error_msg.lower():
            error_type = "server_error"
        elif "memory" in error_msg.lower():
            error_type = "memory_error"

        error_summary[error_type] = error_summary.get(error_type, 0) + 1
