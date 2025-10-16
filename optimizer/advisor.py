from __future__ import annotations

import logging
from typing import Dict, List, Optional

from llm_optimizer.common import (
    ModelConfig,
    calculate_min_tensor_parallel_size,
    calculate_model_memory_bytes,
    get_model_config_and_precision_from_hf,
)
from llm_optimizer.performance import (
    calculate_concurrency_limits,
    find_best_performance,
    find_optimal_concurrency_threshold,
)
from llm_optimizer.predefined.gpus import get_gpu_specs, list_available_gpus
from llm_optimizer.resources.gpu_manager import GPUResourceManager
from llm_optimizer.resources.memory_calculator import ModelMemoryCalculator

from .models import (
    GPURecommendation,
    LLMGPUAnalysis,
    OptimalConfig,
    SystemCompatibility,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_LEN = 1024
DEFAULT_OUTPUT_LEN = 512
GPUS_TO_EVALUATE = (1, 2, 4, 8)


class LLMGPUAdvisor:
    def __init__(self, default_input_len: int = DEFAULT_INPUT_LEN, default_output_len: int = DEFAULT_OUTPUT_LEN):
        self.gpu_manager = GPUResourceManager()
        self.memory_calculator = ModelMemoryCalculator()
        self.available_gpus = list_available_gpus()
        self.default_input_len = default_input_len
        self.default_output_len = default_output_len

    def analyze_llm_for_gpu_recommendation(
        self,
        model_id: str,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
    ) -> LLMGPUAnalysis:
        model_config = get_model_config_and_precision_from_hf(model_id)
        recommendations = self._calculate_gpu_recommendations(model_config)
        if not recommendations:
            raise ValueError("No viable GPU configurations were found for this model.")

        best_gpu = recommendations[0]
        system_compatibility = self._check_system_compatibility(model_config, best_gpu.gpu_name, best_gpu.num_gpus)
        optimal_configs = self._calculate_optimal_configs(
            model_config=model_config,
            gpu_name=best_gpu.gpu_name,
            num_gpus=best_gpu.num_gpus,
            input_len=input_len,
            output_len=output_len,
        )
        performance_analysis = self._analyze_performance(
            model_config=model_config,
            gpu_name=best_gpu.gpu_name,
            num_gpus=best_gpu.num_gpus,
            input_len=input_len,
            output_len=output_len,
        )

        return LLMGPUAnalysis(
            model_config=model_config,
            gpu_recommendations=recommendations,
            system_compatibility=system_compatibility,
            optimal_configs=optimal_configs,
            performance_analysis=performance_analysis,
        )

    def analyze_llm_gpu_combination(
        self,
        model_id: str,
        gpu_name: str,
        num_gpus: int = 1,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
    ) -> LLMGPUAnalysis:
        model_config = get_model_config_and_precision_from_hf(model_id)
        recommendation = self._create_gpu_recommendation(model_config, gpu_name, num_gpus)
        system_compatibility = self._check_system_compatibility(model_config, gpu_name, num_gpus)
        optimal_configs = self._calculate_optimal_configs(
            model_config=model_config,
            gpu_name=gpu_name,
            num_gpus=num_gpus,
            input_len=input_len,
            output_len=output_len,
        )
        performance_analysis = self._analyze_performance(
            model_config=model_config,
            gpu_name=gpu_name,
            num_gpus=num_gpus,
            input_len=input_len,
            output_len=output_len,
        )

        return LLMGPUAnalysis(
            model_config=model_config,
            gpu_recommendations=[recommendation],
            system_compatibility=system_compatibility,
            optimal_configs=optimal_configs,
            performance_analysis=performance_analysis,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _calculate_gpu_recommendations(self, model_config: ModelConfig) -> List[GPURecommendation]:
        recommendations: List[GPURecommendation] = []
        for gpu_name in self.available_gpus:
            try:
                for num_gpus in GPUS_TO_EVALUATE:
                    recommendation = self._create_gpu_recommendation(model_config, gpu_name, num_gpus)
                    if recommendation.suitability_score > 0.3:
                        recommendations.append(recommendation)
            except Exception as exc:
                logger.debug("Skipping GPU %s due to error: %s", gpu_name, exc)

        recommendations.sort(key=lambda rec: rec.suitability_score, reverse=True)
        return recommendations[:5]

    def _create_gpu_recommendation(self, model_config: ModelConfig, gpu_name: str, num_gpus: int) -> GPURecommendation:
        gpu_specs = get_gpu_specs(gpu_name)
        precision = model_config.inferred_precision
        gpu_resources = self.gpu_manager.get_total_resources(num_gpus, gpu_name, precision)

        model_memory_bytes = calculate_model_memory_bytes(model_config, precision)
        model_memory_gb = model_memory_bytes / 1024**3

        suitability_score = self._calculate_suitability_score(
            model_config=model_config,
            gpu_specs=gpu_specs,
            num_gpus=num_gpus,
            model_memory_gb=model_memory_gb,
        )
        reasoning = self._generate_reasoning(model_config, gpu_specs, num_gpus, model_memory_gb)

        return GPURecommendation(
            gpu_name=gpu_name,
            num_gpus=num_gpus,
            total_tflops=gpu_resources.total_tflops,
            total_memory_gb=gpu_resources.total_memory_bytes / 1024**3,
            memory_bandwidth_gbs=gpu_specs["Memory_Bandwidth_GBs"],
            architecture=gpu_specs["Architecture"],
            suitability_score=suitability_score,
            reasoning=reasoning,
        )

    def _calculate_suitability_score(
        self,
        model_config: ModelConfig,
        gpu_specs: Dict[str, float],
        num_gpus: int,
        model_memory_gb: float,
    ) -> float:
        total_memory_gb = gpu_specs["VRAM_GB"] * num_gpus
        memory_ratio = model_memory_gb / total_memory_gb if total_memory_gb else 1.0
        score = 0.0

        if memory_ratio <= 0.3:
            score += 0.4
        elif memory_ratio <= 0.5:
            score += 0.3
        elif memory_ratio <= 0.7:
            score += 0.2
        elif memory_ratio <= 0.9:
            score += 0.1

        tflops = self._precision_tflops(gpu_specs, model_config.inferred_precision)
        if tflops >= 1000:
            score += 0.3
        elif tflops >= 500:
            score += 0.25
        elif tflops >= 300:
            score += 0.2
        elif tflops >= 100:
            score += 0.1

        bandwidth = gpu_specs["Memory_Bandwidth_GBs"]
        if bandwidth >= 3000:
            score += 0.2
        elif bandwidth >= 2000:
            score += 0.15
        elif bandwidth >= 1000:
            score += 0.1
        else:
            score += 0.05

        if num_gpus == 1 and model_memory_gb <= gpu_specs["VRAM_GB"] * 0.8:
            score += 0.1
        elif num_gpus > 1 and model_memory_gb > gpu_specs["VRAM_GB"] * 0.5:
            score += 0.1

        return min(1.0, score)

    def _generate_reasoning(
        self,
        model_config: ModelConfig,
        gpu_specs: Dict[str, float],
        num_gpus: int,
        model_memory_gb: float,
    ) -> str:
        total_memory = gpu_specs["VRAM_GB"] * num_gpus
        memory_ratio = model_memory_gb / total_memory if total_memory else 1.0

        parts: List[str] = []
        if memory_ratio <= 0.3:
            parts.append("excellent memory headroom")
        elif memory_ratio <= 0.5:
            parts.append("comfortable memory usage")
        elif memory_ratio <= 0.7:
            parts.append("acceptable memory usage")
        else:
            parts.append("tight memory budget")

        tflops = self._precision_tflops(gpu_specs, model_config.inferred_precision)
        if tflops >= 500:
            parts.append("high compute capacity")
        elif tflops >= 300:
            parts.append("moderate compute capacity")
        else:
            parts.append("entry-level compute capacity")

        architecture = gpu_specs["Architecture"]
        if architecture in {"Hopper", "Blackwell"}:
            parts.append(f"{architecture} generation GPU")
        elif architecture == "Ampere":
            parts.append("proven Ampere generation GPU")

        return ", ".join(parts)

    def _check_system_compatibility(self, model_config: ModelConfig, gpu_name: str, num_gpus: int) -> SystemCompatibility:
        issues: List[str] = []
        recommendations: List[str] = []

        is_compatible = False
        model_memory_gb = 0.0
        total_memory_gb = 0.0
        min_tp_size = 1
        model_fits = False

        try:
            gpu_specs = get_gpu_specs(gpu_name)
            precision = model_config.inferred_precision

            model_memory_bytes = calculate_model_memory_bytes(model_config, precision)
            model_memory_gb = model_memory_bytes / 1024**3

            gpu_memory_gb = gpu_specs["VRAM_GB"]
            total_memory_gb = gpu_memory_gb * num_gpus
            min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)

            model_fits = model_memory_gb <= total_memory_gb * 0.9
            gpu_count_sufficient = num_gpus >= min_tp_size

            if not model_fits:
                issues.append(
                    f"Model requires {model_memory_gb:.1f} GB but only {total_memory_gb:.1f} GB is available."
                )
                recommendations.append("Increase the GPU count or switch to GPUs with more memory.")

            if not gpu_count_sufficient:
                issues.append(f"At least {min_tp_size} GPU(s) are required for tensor parallelism.")
                recommendations.append("Increase the GPU count to satisfy tensor parallel requirements.")

            if precision == "fp8" and gpu_specs.get("FP8_TFLOPS") is None:
                issues.append(f"{gpu_name} does not provide native FP8 support.")
                recommendations.append("Switch to fp16/bf16 precision or choose an FP8-capable GPU.")

            is_compatible = model_fits and gpu_count_sufficient and not issues
            if is_compatible:
                recommendations.append("The system is ready for the suggested configuration.")

        except Exception as exc:
            issues.append(f"System compatibility check failed: {exc}")
            is_compatible = False
            model_fits = False

        return SystemCompatibility(
            is_compatible=is_compatible,
            model_fits=model_fits,
            min_tp_size=min_tp_size,
            available_gpus=num_gpus,
            memory_requirement_gb=model_memory_gb,
            gpu_memory_gb=total_memory_gb,
            issues=issues,
            recommendations=recommendations,
        )

    def _calculate_optimal_configs(
        self,
        model_config: ModelConfig,
        gpu_name: str,
        num_gpus: int,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
    ) -> List[OptimalConfig]:
        input_len = input_len or self.default_input_len
        output_len = output_len or self.default_output_len
        precision = model_config.inferred_precision

        try:
            best_configs = find_best_performance(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                input_length=input_len,
                output_length=output_len,
            )
            concurrency_limits = calculate_concurrency_limits(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                input_length=input_len,
                output_length=output_len,
            )
            optimal_concurrency = find_optimal_concurrency_threshold(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                input_length=input_len,
                output_length=output_len,
            )
        except Exception as exc:
            logger.debug("Unable to derive optimal configs for %s: %s", gpu_name, exc)
            return []

        throughput = best_configs.get("best_output_throughput")
        if not throughput:
            return []

        configs: List[OptimalConfig] = []
        overall_limit = concurrency_limits.get("overall_limit", optimal_concurrency)

        configs.append(
            OptimalConfig(
                framework="sglang",
                tensor_parallel_size=1,
                data_parallel_size=num_gpus,
                max_concurrent_requests=overall_limit,
                optimal_concurrency=optimal_concurrency,
                precision=precision,
                memory_utilization=0.9,
                estimated_throughput=throughput.output_throughput_tps,
                estimated_latency=throughput.ttft_ms,
                server_args={
                    "tp_size": 1,
                    "dp_size": num_gpus,
                    "max_running_requests": overall_limit,
                    "chunked_prefill_size": 2048,
                    "schedule_conservativeness": 0.6,
                },
                client_args={
                    "max_concurrency": optimal_concurrency,
                    "num_prompts": max(1000, optimal_concurrency * 2),
                },
            )
        )

        configs.append(
            OptimalConfig(
                framework="vllm",
                tensor_parallel_size=1,
                data_parallel_size=num_gpus,
                max_concurrent_requests=overall_limit,
                optimal_concurrency=optimal_concurrency,
                precision=precision,
                memory_utilization=0.9,
                estimated_throughput=throughput.output_throughput_tps,
                estimated_latency=throughput.ttft_ms,
                server_args={
                    "tensor_parallel_size": 1,
                    "data_parallel_size": num_gpus,
                    "max_num_seqs": overall_limit,
                    "max_num_batched_tokens": 16_384,
                },
                client_args={
                    "max_concurrency": optimal_concurrency,
                    "num_prompts": max(1000, optimal_concurrency * 2),
                },
            )
        )

        return configs

    def _analyze_performance(
        self,
        model_config: ModelConfig,
        gpu_name: str,
        num_gpus: int,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
    ) -> Dict[str, object]:
        input_len = input_len or self.default_input_len
        output_len = output_len or self.default_output_len
        precision = model_config.inferred_precision

        try:
            best_configs = find_best_performance(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                input_length=input_len,
                output_length=output_len,
            )
            concurrency_limits = calculate_concurrency_limits(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                input_length=input_len,
                output_length=output_len,
            )
        except Exception as exc:
            logger.debug("Performance analysis failed for %s: %s", gpu_name, exc)
            return {"error": str(exc)}

        return {
            "best_latency": self._serialize_performance(best_configs.get("best_latency")),
            "best_throughput": self._serialize_performance(best_configs.get("best_output_throughput")),
            "concurrency_limits": concurrency_limits,
        }

    @staticmethod
    def _serialize_performance(result) -> Optional[Dict[str, float]]:
        if result is None:
            return None
        return {
            "ttft_ms": result.ttft_ms,
            "itl_ms": result.itl_ms,
            "output_throughput_tps": result.output_throughput_tps,
            "input_throughput_tps": result.input_throughput_tps,
            "concurrency": result.concurrency,
        }

    @staticmethod
    def _precision_tflops(gpu_specs: Dict[str, float], precision: str) -> float:
        if precision in {"fp16", "bf16"}:
            return gpu_specs.get("FP16_TFLOPS", 0.0)
        if precision == "fp8":
            return gpu_specs.get("FP8_TFLOPS", 0.0) or 0.0
        return 0.0
