import asyncio
import json
from typing import Any, Optional, Union, Literal, AsyncGenerator
from pathlib import Path

import httpx
from loguru import logger
from pymongo import MongoClient
from datasets import Dataset

from guidellm.backend import Backend
from guidellm.backend.openai import OpenAIHTTPBackend, StreamingTextResponse, ResponseSummary
from guidellm.benchmark import benchmark_generative_text
from guidellm.dataset import InMemoryDatasetCreator

# Import local mongo client
from mongo_client import Mongo
# Import tokenizer mappings
from tokenizer_utils import OLLAMA_TO_HF_TOKENIZER


class CustomOpenAIBackend(OpenAIHTTPBackend):
    """Extended OpenAI backend that supports custom endpoint paths"""
    
    def __init__(
        self,
        target: Optional[str] = None,
        text_completions_path: Optional[str] = None,
        chat_completions_path: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[float] = None,
        http2: Optional[bool] = True,
        max_output_tokens: Optional[int] = None,
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
    
    def _extract_ollama_delta_content(self, data: dict) -> Optional[str]:
        """Extract streaming delta content from Ollama response format."""
        # Ollama uses "response" field for streaming content
        return data.get("response", "")
    
    async def _iterative_completions_request(
        self,
        type_: Literal["text_completions", "chat_completions"],
        request_id: Optional[str],
        request_prompt_tokens: Optional[int],
        request_output_tokens: Optional[int],
        headers: dict,
        params: dict,
        payload: dict,
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        # Import necessary modules for the implementation
        import time
        import json
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

        logger.info(
            "{} making request: {} to target: {} using http2: {} for "
            "timeout: {} with headers: {} and payload: {}",
            self.__class__.__name__,
            request_id,
            target,
            self.http2,
            self.timeout,
            headers,
            payload,
        )

        response_value = ""
        response_prompt_count: Optional[int] = None
        response_output_count: Optional[int] = None
        iter_count = 0
        start_time = time.time()
        iter_time = start_time
        first_iter_time: Optional[float] = None
        last_iter_time: Optional[float] = None

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
                logger.debug(
                    "{} request: {} received iter response line: {}",
                    self.__class__.__name__,
                    request_id,
                    line,
                )

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
                        
                except json.JSONDecodeError as e:
                    logger.warning(
                        "{} request: {} failed to parse line as JSON: {} - Error: {}",
                        self.__class__.__name__,
                        request_id,
                        line,
                        e,
                    )
                    continue

        logger.info(
            "{} request: {} completed with: {}",
            self.__class__.__name__,
            request_id,
            response_value,
        )

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
    
    def __init__(self, mongo_url: str, db_name: str = "dria_benchmark", collection_name: str = "benchmark_test_data"):
        self.mongo = Mongo(mongo_url)
        self.db_name = db_name
        self.collection_name = collection_name
    
    def load_prompts(self, query: Optional[dict] = None, limit: Optional[int] = None) -> list[dict]:
        """Load prompts from MongoDB"""
        query = query or {}
        cursor = self.mongo.find_many(self.collection_name, query)
        
        if limit:
            cursor = cursor.limit(limit)
        
        prompts = []
        for doc in cursor:
            # Handle different prompt formats
            if 'prompt' in doc:
                prompts.append({'prompt': doc['prompt']})
        
        return prompts
    
    def create_dataset(self, query: Optional[dict] = None, limit: Optional[int] = None) -> Dataset:
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
            random_seed=42
        )
        
        return dataset


class GuideLLMBenchmarkClient:
    """Main benchmark client that integrates guidellm with custom configuration"""
    
    # Add tokenizer mapping as class attribute
    OLLAMA_TO_HF_TOKENIZER = OLLAMA_TO_HF_TOKENIZER
    
    def __init__(
        self,
        base_url: str,
        model: str,
        mongo_url: Optional[str] = None,
        text_completions_path: Optional[str] = None,
        chat_completions_path: Optional[str] = None,
        api_key: Optional[str] = None,
        max_output_tokens: Optional[int] = 100,
        processor: Optional[str] = None,  # Allow override of processor/tokenizer
    ):
        self.base_url = base_url
        self.model = model
        self.mongo_url = mongo_url
        self.text_completions_path = text_completions_path or "/api/generate"
        self.chat_completions_path = chat_completions_path or "/v1/chat/completions"
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.processor = processor  # Store custom processor
        
        # Initialize MongoDB loader if URL provided
        self.mongo_loader = MongoDatasetLoader(mongo_url) if mongo_url else None
    
    async def run_benchmark(
        self,
        data: Optional[Union[str, Path, list]] = None,
        mongo_query: Optional[dict] = None,
        mongo_limit: Optional[int] = None,
        rate_type: str = "synchronous",
        rate: Optional[Union[int, float, list]] = None,
        max_seconds: Optional[float] = None,
        max_requests: Optional[int] = None,
        warmup_percent: Optional[float] = 0.1,
        output_path: Optional[Union[str, Path]] = None,
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
            logger.info(f"Using tokenizer: {processor} for model: {self.model}")
        else:
            # Default to None to avoid errors with unknown models
            processor = None
            logger.warning(
                f"No tokenizer mapping found for model '{self.model}'. "
                "Token counting will be disabled. To enable token counting, "
                "provide a 'processor' parameter with a valid HuggingFace tokenizer name."
            )
        
        # Test tokenizer loading if one is specified
        if processor and not disable_token_counting:
            try:
                from guidellm.utils.hf_transformers import check_load_processor
                # Test load the processor to ensure it works
                test_processor = check_load_processor(
                    processor,
                    processor_args=None,
                    error_msg="Testing tokenizer availability"
                )
                logger.info(f"Successfully validated tokenizer: {processor}")
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer '{processor}' for model '{self.model}': {e}. "
                    "Disabling token counting for this benchmark run."
                )
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


