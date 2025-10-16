from dataclasses import dataclass
from typing import Any, Dict, List

from llm_optimizer.common import ModelConfig


@dataclass
class GPURecommendation:
    gpu_name: str
    num_gpus: int
    total_tflops: float
    total_memory_gb: float
    memory_bandwidth_gbs: float
    architecture: str
    suitability_score: float
    reasoning: str


@dataclass
class SystemCompatibility:
    is_compatible: bool
    model_fits: bool
    min_tp_size: int
    available_gpus: int
    memory_requirement_gb: float
    gpu_memory_gb: float
    issues: List[str]
    recommendations: List[str]


@dataclass
class OptimalConfig:
    framework: str
    tensor_parallel_size: int
    data_parallel_size: int
    max_concurrent_requests: int
    optimal_concurrency: int
    precision: str
    memory_utilization: float
    estimated_throughput: float
    estimated_latency: float
    server_args: Dict[str, Any]
    client_args: Dict[str, Any]


@dataclass
class LLMGPUAnalysis:
    model_config: ModelConfig
    gpu_recommendations: List[GPURecommendation]
    system_compatibility: SystemCompatibility
    optimal_configs: List[OptimalConfig]
    performance_analysis: Dict[str, Any]


__all__ = [
    "GPURecommendation",
    "SystemCompatibility",
    "OptimalConfig",
    "LLMGPUAnalysis",
]
