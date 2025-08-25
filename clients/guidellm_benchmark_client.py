import json
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
from guidellm.backend.response import RequestArgs
from guidellm.benchmark import benchmark_generative_text
from guidellm.dataset import InMemoryDatasetCreator

from clients.mongo_client import Mongo
from utils.ollama_utils import HF_TOKENIZER


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

        self._text_completions_path = text_completions_path or "/v1/completions"
        self._chat_completions_path = chat_completions_path or "/v1/chat/completions"

    def _extract_ollama_delta_content(self, data: dict) -> str | None:
        return data.get("response", "")

    async def _iterative_completions_request(
        self,
        type_: Literal["text_completions", "chat_completions"],
        request_id: str | None,
        request_prompt_tokens: int | None,
        request_output_tokens: int | None,
        headers: dict,
        payload: dict,
        params: dict | None = None,
    ) -> AsyncGenerator[StreamingTextResponse | ResponseSummary, None]:
        if params is None:
            params = {}

        # Use custom paths for requests
        if type_ == "text_completions":
            target = f"{self.target}{self._text_completions_path}"
        elif type_ == "chat_completions":
            target = f"{self.target}{self._chat_completions_path}"
        payload = payload.copy()
        payload["stream"] = True

        # Remove unsupported parameters for SGlang,Vllm and Ollama
        unsupported_params = ["stream_options", "max_completion_tokens", "ignore_eos"]
        for param in unsupported_params:
            payload.pop(param, None)

        if type_ == "text_completions" and "prompt" not in payload:
            if "messages" in payload:
                payload["prompt"] = payload.pop("messages")[-1].get("content", "")

        # Set Variables for measuring performance
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


# Register the custom backend for guidellm
Backend._registry["custom_openai"] = CustomOpenAIBackend


class MongoDatasetLoader:
    """For loading prompts from MongoDB and converting to guidellm dataset"""

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
    """

    # Add tokenizer mapping as class attribute
    HF_TOKENIZER = HF_TOKENIZER

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
        # Map Ollama model to HF tokenizer if model name contains ":" else use model name for VLLM and SGlang
        self.processor = HF_TOKENIZER[model] if ":" in model else model

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.health_check_timeout = health_check_timeout

        self.mongo_loader = MongoDatasetLoader(mongo_url) if mongo_url else None
        self.mongo_client = Mongo(mongo_url) if mongo_url else None

    def _calculate_backoff_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)

    @contextmanager
    def _timeout_context(self, timeout: float):
        """To calculate timer on timeout"""

        def timeout_handler():
            # We don't do anything here RN.
            pass

        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def health_check(self) -> dict[str, Any]:
        """To check if the model server is healthy"""

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

                # Try to get model list from server to check server and model availability
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

    async def run_benchmark(
        self,
        data: str | Path | list | None = None,
        mongo_query: dict | None = None,
        mongo_limit: int | None = None,
        rate_type: str = "synchronous",
        rate: float | None = None,
        max_seconds: float | None = None,
        max_requests: int | None = None,
        warmup_percent: float | None = 0.1,
        output_path: str | Path | None = None,
        show_progress: bool = True,
        disable_token_counting: bool = False,
    ):
        """Run benchmark with custom configuration"""

        # Prepare data
        if self.mongo_loader and (mongo_query is not None or data is None):
            # Load from MongoDB
            dataset = self.mongo_loader.create_dataset(mongo_query, mongo_limit)
            data_source = dataset
        else:
            # Use provided data without mongo
            data_source = data or []

        # Create backend arguments
        backend_args = {
            "text_completions_path": self.text_completions_path,
            "chat_completions_path": self.chat_completions_path,
            "api_key": self.api_key,
            "max_output_tokens": self.max_output_tokens,
        }

        # Determine processor/tokenizer if sent it true then disable token counting
        if disable_token_counting:
            processor = None
        else:
            processor = self.processor

        # Handle rate type
        if isinstance(rate, int):
            rate = float(rate)

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

    def _extract_benchmark_metrics(
        self, benchmark_report, pod_id: str, rate: int | None, test_type: str = None
    ) -> dict:
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
        if (
            benchmark_report.run_stats.end_time
            and benchmark_report.run_stats.start_time
        ):
            result["benchmark_duration"] = (
                benchmark_report.run_stats.end_time
                - benchmark_report.run_stats.start_time
            )

        # Handle rate extraction for profile types
        if not rate and self.mongo_client:
            result["rate"] = self.mongo_client.get_metric(
                benchmark_report.args.profile, "streams[0]"
            )

        return result
