import time
import random
import requests
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

def wait_for_server_ready(
    url: str, 
    endpoint: str = "/health", 
    max_wait: int = 600,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    timeout: int = 10,
    expected_status: int = 200,
    custom_check: Optional[Callable[[requests.Response], bool]] = None
) -> bool:
    """
    Wait for server to be ready using exponential backoff with jitter.
    
    Args:
        url: Base URL of the server
        endpoint: Health check endpoint
        max_wait: Maximum time to wait in seconds
        initial_delay: Initial delay between attempts
        max_delay: Maximum delay between attempts
        backoff_factor: Exponential backoff factor
        timeout: Request timeout in seconds
        expected_status: Expected HTTP status code
        custom_check: Custom function to validate response
        
    Returns:
        True if server is ready, False if timeout
    """
    start_time = time.time()
    attempt = 0
    full_url = f"{url.rstrip('/')}{endpoint}"
    
    logger.info(f"Waiting for server to be ready at {full_url}")
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(full_url, timeout=timeout)
            
            # Check status code
            if response.status_code == expected_status:
                # Custom validation if provided
                if custom_check:
                    if custom_check(response):
                        logger.info(f"Server ready after {time.time() - start_time:.2f}s (attempt {attempt + 1})")
                        return True
                else:
                    logger.info(f"Server ready after {time.time() - start_time:.2f}s (attempt {attempt + 1})")
                    return True
            
            logger.debug(f"Server not ready (status: {response.status_code}), attempt {attempt + 1}")
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Connection failed (attempt {attempt + 1}): {e}")
        
        # Exponential backoff with jitter
        delay = min(max_delay, initial_delay * (backoff_factor ** attempt))
        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
        total_delay = delay + jitter
        
        logger.debug(f"Waiting {total_delay:.2f}s before next attempt")
        time.sleep(total_delay)
        attempt += 1
    
    logger.error(f"Server not ready after {max_wait}s timeout")
    return False

def wait_for_ollama_ready(url: str, max_wait: int = 600) -> bool:
    """Wait for Ollama server to be ready."""
    def ollama_check(response):
        try:
            return "Ollama is running" in response.text
        except:
            return False
    
    return wait_for_server_ready(
        url=url,
        endpoint="/",
        max_wait=max_wait,
        custom_check=ollama_check
    )

def wait_for_vllm_ready(url: str, max_wait: int = 600) -> bool:
    """Wait for vLLM server to be ready."""
    def vllm_check(response):
        try:
            # vLLM health endpoint should return JSON
            data = response.json()
            return data.get("status") == "ok" or "model" in data
        except:
            return False
    
    return wait_for_server_ready(
        url=url,
        endpoint="/health",
        max_wait=max_wait,
        custom_check=vllm_check
    )

def wait_for_sglang_ready(url: str, max_wait: int = 600) -> bool:
    """Wait for SGLang server to be ready."""
    def sglang_check(response):
        try:
            data = response.json()
            return "model_name" in data or data.get("status") == "ready"
        except:
            return False
    
    return wait_for_server_ready(
        url=url,
        endpoint="/get_model_info",
        max_wait=max_wait,
        custom_check=sglang_check
    ) 