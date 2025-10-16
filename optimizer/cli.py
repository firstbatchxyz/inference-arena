from __future__ import annotations

import argparse
import sys
from typing import Iterable

from llm_optimizer.predefined.gpus import get_gpu_specs

from .advisor import LLMGPUAdvisor
from .reporting import format_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggest GPUs and serving configs for large language models.")
    subcommands = parser.add_subparsers(dest="command")

    recommend = subcommands.add_parser(
        "recommend",
        help="Recommend GPUs for a given Hugging Face model.",
    )
    recommend.add_argument("model_id", help="Hugging Face model identifier.")
    recommend.add_argument("--input-len", type=int, default=None, help="Input token length used for estimation.")
    recommend.add_argument("--output-len", type=int, default=None, help="Output token length used for estimation.")

    analyze = subcommands.add_parser(
        "analyze",
        help="Evaluate a specific model and GPU combination.",
    )
    analyze.add_argument("model_id", help="Hugging Face model identifier.")
    analyze.add_argument("gpu_name", help="GPU model name, e.g. A100 or H100.")
    analyze.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs.")
    analyze.add_argument("--input-len", type=int, default=None, help="Input token length used for estimation.")
    analyze.add_argument("--output-len", type=int, default=None, help="Output token length used for estimation.")

    subcommands.add_parser("list-gpus", help="List supported GPU profiles.")

    return parser


def handle_recommend(advisor: LLMGPUAdvisor, args: argparse.Namespace) -> int:
    analysis = advisor.analyze_llm_for_gpu_recommendation(
        model_id=args.model_id,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    print(format_analysis(analysis))
    return 0


def handle_analyze(advisor: LLMGPUAdvisor, args: argparse.Namespace) -> int:
    analysis = advisor.analyze_llm_gpu_combination(
        model_id=args.model_id,
        gpu_name=args.gpu_name,
        num_gpus=args.num_gpus,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    print(format_analysis(analysis))
    return 0


def handle_list_gpus(advisor: LLMGPUAdvisor) -> int:
    lines = ["Supported GPUs:"]
    for gpu in advisor.available_gpus:
        try:
            specs = get_gpu_specs(gpu)
            lines.append(
                f"  - {gpu}: {specs['Architecture']} | {specs['VRAM_GB']} GB | {specs['FP16_TFLOPS']} FP16 TFLOPS"
            )
        except Exception:
            lines.append(f"  - {gpu}")
    print("\n".join(lines))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None:
        parser.print_help()
        return 1

    advisor = LLMGPUAdvisor()

    if args.command == "recommend":
        return handle_recommend(advisor, args)
    if args.command == "analyze":
        return handle_analyze(advisor, args)
    if args.command == "list-gpus":
        return handle_list_gpus(advisor)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
