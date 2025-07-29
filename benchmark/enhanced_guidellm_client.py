"""
Enhanced GuideLLM Benchmark Client with Robust Error Handling

This module extends the existing GuideLLM client with:
- Exponential backoff retry mechanisms
- Connection timeout handling
- Graceful failure recovery
- Detailed logging and monitoring
"""

import asyncio
import time
import json
from typing import Any, Optional, Union, Literal, AsyncGenerator, Dict, List
from pathlib import Path
import random

import httpx
from loguru import logger
import threading
from contextlib import contextmanager

from guidellm_benchmark_client import GuideLLMBenchmarkClient
from mongo_client import Mongo


class EnhancedGuideLLMBenchmarkClient(GuideLLMBenchmarkClient):
    """Enhanced benchmark client with robust error handling and retry mechanisms."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        mongo_url: str,
        text_completions_path: str = "/v1/completions",
        processor: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 300.0,
        health_check_timeout: float = 30.0
    ):
        """
        Initialize enhanced client with retry and timeout configurations.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            timeout: Request timeout (seconds)
            health_check_timeout: Health check timeout (seconds)
        """
        super().__init__(
            base_url=base_url,
            model=model,
            mongo_url=mongo_url,
            text_completions_path=text_completions_path,
            processor=processor
        )
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.health_check_timeout = health_check_timeout
        self.mongo_client = Mongo(mongo_url)
        
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)
    
    @contextmanager
    def _timeout_context(self, timeout: float):
        """Context manager for operation timeouts."""
        def timeout_handler():
            logger.warning(f"Operation timed out after {timeout} seconds")
            
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the model server.
        
        Returns:
            Dict with health status, response time, and server info
        """
        health_status = {
            'healthy': False,
            'response_time_ms': None,
            'server_info': None,
            'error': None,
            'timestamp': time.time()
        }
        
        try:
            with self._timeout_context(self.health_check_timeout):
                start_time = time.time()
                
                # Try to get model list
                response = httpx.get(
                    f"{self.base_url}/v1/models",
                    timeout=self.health_check_timeout
                )
                
                response_time = (time.time() - start_time) * 1000
                health_status['response_time_ms'] = response_time
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        model_found = any(
                            model.get('id') == self.model 
                            for model in data['data']
                        )
                        
                        if model_found:
                            health_status['healthy'] = True
                            health_status['server_info'] = {
                                'available_models': [m.get('id') for m in data['data']],
                                'target_model': self.model,
                                'response_code': response.status_code
                            }
                        else:
                            health_status['error'] = f"Model {self.model} not found in available models"
                    else:
                        health_status['error'] = "No models available on server"
                else:
                    health_status['error'] = f"Server returned status {response.status_code}"
                    
        except httpx.TimeoutException:
            health_status['error'] = f"Health check timed out after {self.health_check_timeout}s"
        except httpx.ConnectError:
            health_status['error'] = "Cannot connect to server"
        except Exception as e:
            health_status['error'] = f"Health check failed: {str(e)}"
        
        logger.info(f"Health check result: {health_status}")
        return health_status
    
    def wait_for_server_ready(self, max_wait_time: float = 600.0, check_interval: float = 10.0) -> bool:
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
        
        logger.info(f"Waiting for server {self.base_url} to be ready...")
        
        while time.time() - start_time < max_wait_time:
            health = self.health_check()
            
            if health['healthy']:
                logger.info(f"Server ready after {time.time() - start_time:.1f}s")
                return True
            
            # Calculate next check interval with backoff
            wait_time = min(check_interval * (1.5 ** attempt), 60.0)
            logger.info(f"Server not ready (attempt {attempt + 1}): {health['error']}. Waiting {wait_time:.1f}s...")
            
            time.sleep(wait_time)
            attempt += 1
        
        logger.error(f"Server failed to become ready within {max_wait_time}s")
        return False
    
    def run_sync_with_retry(
        self,
        max_seconds: Optional[int] = None,
        mongo_query: Optional[Dict] = None,
        rate_type: str = "concurrent",
        output_path: str = "benchmark_results.json",
        rate: Optional[int] = None,
        **kwargs
    ) -> tuple:
        """
        Run benchmark with retry mechanism and error handling.
        
        Returns:
            Tuple of (report, path) or (None, None) if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Starting benchmark attempt {attempt + 1}/{self.max_retries + 1}")
                
                # Health check before running benchmark
                if not self.health_check()['healthy']:
                    raise RuntimeError("Server health check failed before benchmark")
                
                # Run the actual benchmark
                result = self.run_sync(
                    max_seconds=max_seconds,
                    mongo_query=mongo_query or {},
                    rate_type=rate_type,
                    output_path=output_path,
                    rate=rate,
                    **kwargs
                )
                
                logger.info(f"Benchmark completed successfully on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Benchmark attempt {attempt + 1} failed: {str(e)}")
                
                # Log the failure to database
                self._log_benchmark_failure(attempt + 1, str(e))
                
                # Don't retry on the last attempt
                if attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} benchmark attempts failed")
        
        # Log final failure
        self._log_benchmark_final_failure(str(last_exception))
        return None, None
    
    def run_benchmark_suite(
        self,
        pod_id: str,
        concurrent_rates: List[int] = None,
        include_throughput: bool = True,
        max_seconds_per_test: int = 30
    ) -> Dict[str, Any]:
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
            'pod_id': pod_id,
            'start_time': time.time(),
            'concurrent_results': [],
            'throughput_result': None,
            'failures': [],
            'total_tests': len(concurrent_rates) + (1 if include_throughput else 0),
            'successful_tests': 0,
            'error_summary': {}
        }
        
        # Test concurrent rates
        for rate in concurrent_rates:
            logger.info(f"Running concurrent benchmark with rate={rate}")
            
            try:
                report, path = self.run_sync_with_retry(
                    max_seconds=max_seconds_per_test,
                    rate_type="concurrent",
                    rate=rate,
                    output_path=f"benchmark_results_concurrent_{rate}.json"
                )
                
                if report and report.benchmarks:
                    benchmark_report = report.benchmarks[0]
                    
                    result_data = self._extract_benchmark_metrics(benchmark_report, pod_id, rate)
                    suite_results['concurrent_results'].append(result_data)
                    suite_results['successful_tests'] += 1
                    
                    # Log to database
                    self.mongo_client.insert_one("benchmark_results", result_data)
                    
                else:
                    suite_results['failures'].append({
                        'test_type': 'concurrent',
                        'rate': rate,
                        'error': 'No benchmark results returned'
                    })
                    
            except Exception as e:
                error_msg = str(e)
                suite_results['failures'].append({
                    'test_type': 'concurrent',
                    'rate': rate,
                    'error': error_msg
                })
                self._track_error_type(suite_results['error_summary'], error_msg)
            
            # Brief pause between tests
            time.sleep(2)
        
        # Test throughput
        if include_throughput:
            logger.info("Running throughput benchmark")
            
            try:
                report, path = self.run_sync_with_retry(
                    max_seconds=max_seconds_per_test,
                    rate_type="throughput",
                    output_path="benchmark_results_throughput.json"
                )
                
                if report and report.benchmarks:
                    benchmark_report = report.benchmarks[0]
                    
                    result_data = self._extract_benchmark_metrics(benchmark_report, pod_id, None, "throughput")
                    suite_results['throughput_result'] = result_data
                    suite_results['successful_tests'] += 1
                    
                    # Log to database
                    self.mongo_client.insert_one("benchmark_results", result_data)
                    
                else:
                    suite_results['failures'].append({
                        'test_type': 'throughput',
                        'error': 'No benchmark results returned'
                    })
                    
            except Exception as e:
                error_msg = str(e)
                suite_results['failures'].append({
                    'test_type': 'throughput',
                    'error': error_msg
                })
                self._track_error_type(suite_results['error_summary'], error_msg)
        
        suite_results['end_time'] = time.time()
        suite_results['total_duration'] = suite_results['end_time'] - suite_results['start_time']
        suite_results['success_rate'] = suite_results['successful_tests'] / suite_results['total_tests']
        
        logger.info(f"Benchmark suite completed: {suite_results['successful_tests']}/{suite_results['total_tests']} tests successful")
        
        return suite_results
    
    def _extract_benchmark_metrics(self, benchmark_report, pod_id: str, rate: Optional[int], test_type: str = None) -> Dict:
        """Extract metrics from benchmark report."""
        return {
            "pod_id": pod_id,
            "benchmark_type": test_type or benchmark_report.args.profile.type_,
            "rate": rate or self.mongo_client.safe_get_metric(benchmark_report.args.profile, "streams[0]"),
            "max_number": benchmark_report.args.max_number,
            "warmup_number": benchmark_report.args.warmup_number,
            "benchmark_duration": (
                benchmark_report.run_stats.end_time - benchmark_report.run_stats.start_time
                if benchmark_report.run_stats.end_time and benchmark_report.run_stats.start_time
                else None
            ),
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
            "timestamp": time.time()
        }
    
    def _log_benchmark_failure(self, attempt: int, error: str):
        """Log individual benchmark failure to database."""
        try:
            self.mongo_client.insert_one("benchmark_failures", {
                "base_url": self.base_url,
                "model": self.model,
                "attempt": attempt,
                "error": error,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"Failed to log benchmark failure: {e}")
    
    def _log_benchmark_final_failure(self, error: str):
        """Log final benchmark failure after all retries."""
        try:
            self.mongo_client.insert_one("benchmark_final_failures", {
                "base_url": self.base_url,
                "model": self.model,
                "final_error": error,
                "max_retries": self.max_retries,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"Failed to log final benchmark failure: {e}")
    
    def _track_error_type(self, error_summary: Dict, error_msg: str):
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