import redis
import json
import os
import hashlib
from typing import Any, Optional, Dict, List
from datetime import timedelta, datetime
from functools import wraps
import asyncio
from fastapi import HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            # Synchronous Redis client
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to in-memory cache.")
            self.redis_client = None
            # Fallback to in-memory cache
            self._memory_cache = {}
            self._cache_expiry = {}
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a unique cache key based on function arguments"""
        # Convert arguments to a string representation
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        # Hash the key to ensure consistent length
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime and other objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, BaseModel):
            return obj.model_dump(by_alias=True)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for cache storage"""
        try:
            # If it's a Pydantic model, convert to dict first
            if isinstance(data, BaseModel):
                data = data.model_dump(by_alias=True)
            
            return json.dumps(data, default=self._json_serializer, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            return json.dumps({"error": "serialization_failed"})
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from cache"""
        try:
            return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        """Set a value in cache with TTL"""
        try:
            if self.redis_client:
                serialized_value = self._serialize_data(value)
                result = self.redis_client.setex(key, ttl_seconds, serialized_value)
                return bool(result)
            else:
                # Fallback to in-memory cache
                self._memory_cache[key] = value
                self._cache_expiry[key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
                return True
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(key)
                if cached_data and isinstance(cached_data, str):
                    return self._deserialize_data(cached_data)
                return None
            else:
                # Fallback to in-memory cache
                if key in self._memory_cache:
                    # Check if expired
                    if datetime.utcnow() < self._cache_expiry.get(key, datetime.min):
                        return self._memory_cache[key]
                    else:
                        # Remove expired entry
                        del self._memory_cache[key]
                        del self._cache_expiry[key]
                return None
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                # Fallback to in-memory cache
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    del self._cache_expiry[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern"""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys and isinstance(keys, list):
                    result = self.redis_client.delete(*keys)
                    return int(result) if result else 0
                return 0
            else:
                # Fallback to in-memory cache
                keys_to_delete = [key for key in self._memory_cache.keys() if pattern in key]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    if key in self._cache_expiry:
                        del self._cache_expiry[key]
                return len(keys_to_delete)
        except Exception as e:
            logger.error(f"Cache delete pattern failed for pattern {pattern}: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all cache entries"""
        try:
            if self.redis_client:
                self.redis_client.flushdb()
                return True
            else:
                self._memory_cache.clear()
                self._cache_expiry.clear()
                return True
        except Exception as e:
            logger.error(f"Cache clear all failed: {e}")
            return False

# Global cache instance
cache_service = CacheService()

def cache_result(ttl_seconds: int = 300, cache_prefix: str = "api"):
    """
    Decorator to cache function results
    
    Args:
        ttl_seconds: Cache TTL in seconds (default: 5 minutes)
        cache_prefix: Prefix for cache keys
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_service._generate_cache_key(f"{cache_prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Call the original function
            logger.info(f"Cache miss for {func.__name__}, executing function")
            result = await func(*args, **kwargs)
            
            # Cache the result
            cache_service.set(cache_key, result, ttl_seconds)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_service._generate_cache_key(f"{cache_prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Call the original function
            logger.info(f"Cache miss for {func.__name__}, executing function")
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_service.set(cache_key, result, ttl_seconds)
            
            return result
        
        # Return the appropriate wrapper based on whether the function is async
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def invalidate_cache_pattern(pattern: str):
    """Helper function to invalidate cache entries matching a pattern"""
    return cache_service.delete_pattern(pattern)

def invalidate_cache_key(key: str):
    """Helper function to invalidate a specific cache key"""
    return cache_service.delete(key)

# Cache invalidation helpers for different data types
def invalidate_benchmark_cache(pod_id: Optional[str] = None):
    """Invalidate benchmark-related cache entries"""
    patterns = [
        "api:benchmark_results*",
        "api:get_all_benchmarks*",
        "api:get_benchmark_results_by_filters*",
        "api:multiple_benchmark_results*"
    ]
    
    if pod_id:
        patterns.append(f"*{pod_id}*")
    
    total_deleted = 0
    for pattern in patterns:
        total_deleted += invalidate_cache_pattern(pattern)
    
    logger.info(f"Invalidated {total_deleted} benchmark cache entries")
    return total_deleted

def invalidate_comments_cache(pod_id: Optional[str] = None):
    """Invalidate comment-related cache entries"""
    patterns = [
        "api:get_comments*",
        "api:get_all_comments*",
        "api:get_user_comments*"
    ]
    
    if pod_id:
        patterns.append(f"*{pod_id}*")
    
    total_deleted = 0
    for pattern in patterns:
        total_deleted += invalidate_cache_pattern(pattern)
    
    logger.info(f"Invalidated {total_deleted} comment cache entries")
    return total_deleted

def invalidate_gpu_cache():
    """Invalidate GPU pricing cache entries"""
    patterns = [
        "api:get_gpu_prices_by_platform*",
        "api:get_gpu_prices_by_pod_id*"
    ]
    
    total_deleted = 0
    for pattern in patterns:
        total_deleted += invalidate_cache_pattern(pattern)
    
    logger.info(f"Invalidated {total_deleted} GPU cache entries")
    return total_deleted 