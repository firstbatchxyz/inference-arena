from __future__ import annotations

from typing import List

from .models import LLMGPUAnalysis, OptimalConfig


def format_analysis(analysis: LLMGPUAnalysis) -> str:
    lines: List[str] = []
    model = analysis.model_config

    lines.append("=" * 72)
    lines.append("LLM GPU Analysis")
    lines.append("=" * 72)
    lines.append("")

    lines.append("Model")
    lines.append(f"  Parameters : {model.num_params / 1e9:.1f}B")
    lines.append(f"  Layers     : {model.num_layers}")
    lines.append(f"  Hidden Dim : {model.hidden_dim}")
    lines.append(f"  Precision  : {model.inferred_precision}")
    lines.append("")

    lines.append("Top GPU Candidates")
    for index, rec in enumerate(analysis.gpu_recommendations, start=1):
        lines.append(f"  {index}. {rec.num_gpus}× {rec.gpu_name} ({rec.architecture})")
        lines.append(f"     Score      : {rec.suitability_score:.2f}")
        lines.append(f"     TFLOPS     : {rec.total_tflops:.1f}")
        lines.append(f"     Memory     : {rec.total_memory_gb:.1f} GB")
        lines.append(f"     Bandwidth  : {rec.memory_bandwidth_gbs:.0f} GB/s")
        lines.append(f"     Notes      : {rec.reasoning}")
        lines.append("")

    compatibility = analysis.system_compatibility
    lines.append("System Compatibility")
    lines.append(f"  Compatible        : {'yes' if compatibility.is_compatible else 'no'}")
    lines.append(f"  Model fits        : {'yes' if compatibility.model_fits else 'no'}")
    lines.append(f"  Min tensor parallel: {compatibility.min_tp_size}")
    lines.append(f"  Memory required   : {compatibility.memory_requirement_gb:.1f} GB")
    lines.append(f"  Memory available  : {compatibility.gpu_memory_gb:.1f} GB")

    if compatibility.issues:
        lines.append("  Issues")
        for issue in compatibility.issues:
            lines.append(f"    - {issue}")

    if compatibility.recommendations:
        lines.append("  Recommendations")
        for recommendation in compatibility.recommendations:
            lines.append(f"    - {recommendation}")
    lines.append("")

    if analysis.optimal_configs:
        lines.append("Suggested Serving Configurations")
        for config in analysis.optimal_configs:
            _append_config(lines, config)
    else:
        lines.append("Suggested Serving Configurations")
        lines.append("  Unable to derive serving presets for the current setup.")
    lines.append("")

    perf = analysis.performance_analysis
    lines.append("Performance Summary")
    if isinstance(perf, dict) and "error" in perf:
        lines.append(f"  {perf['error']}")
    else:
        best_latency = perf.get("best_latency")
        if best_latency:
            lines.append("  Best Latency")
            lines.append(f"    TTFT       : {best_latency['ttft_ms']:.1f} ms")
            lines.append(f"    ITL        : {best_latency['itl_ms']:.1f} ms")
            lines.append(f"    Concurrency: {best_latency['concurrency']}")
        best_throughput = perf.get("best_throughput")
        if best_throughput:
            lines.append("  Best Throughput")
            lines.append(f"    Output TPS : {best_throughput['output_throughput_tps']:.1f}")
            lines.append(f"    Input TPS  : {best_throughput['input_throughput_tps']:.1f}")
            lines.append(f"    Concurrency: {best_throughput['concurrency']}")

        concurrency_limits = perf.get("concurrency_limits") or {}
        if concurrency_limits:
            lines.append("  Concurrency Limits")
            for key, value in concurrency_limits.items():
                lines.append(f"    {key}: {value}")

    lines.append("=" * 72)
    return "\n".join(lines)


def _append_config(lines: List[str], config: OptimalConfig) -> None:
    lines.append(f"  {config.framework.upper()}")
    lines.append(f"    TP × DP           : {config.tensor_parallel_size} × {config.data_parallel_size}")
    lines.append(f"    Max requests      : {config.max_concurrent_requests}")
    lines.append(f"    Optimal concurrency: {config.optimal_concurrency}")
    lines.append(f"    Throughput (tokens/s): {config.estimated_throughput:.1f}")
    lines.append(f"    Latency (ms)      : {config.estimated_latency:.1f}")
    lines.append(f"    Precision         : {config.precision}")
    lines.append(f"    Memory util       : {config.memory_utilization:.2f}")
    lines.append("    Server args")
    for key, value in config.server_args.items():
        lines.append(f"      - {key}={value}")
    lines.append("    Client args")
    for key, value in config.client_args.items():
        lines.append(f"      - {key}={value}")
    lines.append("")
