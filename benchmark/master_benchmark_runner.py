#!/usr/bin/env python3
"""
Master Benchmark Runner

This script orchestrates comprehensive benchmarking across all model families
(Gemma, Falcon, Qwen, Llama) and all inference engines (Ollama, vLLM, SGLang)
with intelligent GPU selection and robust error handling.
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the benchmark directory to path for imports
sys.path.append(str(Path(__file__).parent))

from gpu_selector import GPUSelector
from enhanced_guidellm_client import EnhancedGuideLLMBenchmarkClient
from run_ollama_benchmark_with_runpod import create_ollama_pod
from run_vllm_benchmark_with_runpod import create_vllm_pod
from run_sglang_benchmark_with_runpod import create_sglang_pod
from mongo_client import Mongo
from tokenizer_utils import OLLAMA_TO_HF_TOKENIZER


class MasterBenchmarkRunner:
    """Master benchmark orchestrator for all model families and inference engines."""
    
    def __init__(self, 
                 config_path: str = "configs/runpod_gpus.yaml",
                 mongo_url: Optional[str] = None,
                 dry_run: bool = False):
        """
        Initialize the master benchmark runner.
        
        Args:
            config_path: Path to GPU configuration file
            mongo_url: MongoDB connection URL
            dry_run: If True, only plan benchmarks without execution
        """
        self.gpu_selector = GPUSelector(config_path)
        self.mongo_url = mongo_url or os.getenv("MONGODB_URL")
        self.dry_run = dry_run
        self.mongo_client = Mongo(self.mongo_url) if self.mongo_url else None
        
        # Model mappings from Ollama to HuggingFace format
        self.model_mappings = {
            'gemma': {
                '1b': {'ollama_id': 'gemma2:2b', 'hf_id': 'google/gemma-2-2b-it'},
                '4b': {'ollama_id': 'gemma2:9b', 'hf_id': 'google/gemma-2-9b-it'},
                '12b': {'ollama_id': 'gemma2:27b', 'hf_id': 'google/gemma-2-27b-it'}
            },
            'falcon': {
                '10b': {'ollama_id': 'falcon3:10b', 'hf_id': 'tiiuae/falcon-11B'},
                '180b': {'ollama_id': 'falcon:180b', 'hf_id': 'tiiuae/falcon-180B'}
            },
            'qwen': {
                '14b': {'ollama_id': 'qwen2.5:14b', 'hf_id': 'Qwen/Qwen2.5-14B-Instruct'},
                '32b': {'ollama_id': 'qwen2.5:32b', 'hf_id': 'Qwen/Qwen2.5-32B-Instruct'},
                '72b': {'ollama_id': 'qwen2.5:72b', 'hf_id': 'Qwen/Qwen2.5-72B-Instruct'}
            },
            'llama': {
                '70b': {'ollama_id': 'llama3.1:70b', 'hf_id': 'meta-llama/Llama-3.1-70B-Instruct'},
                '405b': {'ollama_id': 'llama3.1:405b', 'hf_id': 'meta-llama/Llama-3.1-405B-Instruct'}
            }
        }
        
        # Inference engines
        self.inference_engines = {
            'ollama': {
                'function': create_ollama_pod,
                'port': 11434,
                'text_completions_path': '/api/generate'
            },
            'vllm': {
                'function': create_vllm_pod,
                'port': 8000,
                'text_completions_path': '/v1/completions'
            },
            'sglang': {
                'function': create_sglang_pod,
                'port': 30000,
                'text_completions_path': '/v1/completions'
            }
        }
        
        # Benchmark execution state
        self.execution_summary = {
            'total_benchmarks': 0,
            'successful_benchmarks': 0,
            'failed_benchmarks': 0,
            'skipped_benchmarks': 0,
            'start_time': None,
            'end_time': None,
            'results': []
        }
    
    def generate_benchmark_plan(self, 
                              prioritize_high_end: bool = True,
                              max_configs_per_model: int = 2,
                              include_engines: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate a comprehensive benchmark execution plan.
        
        Args:
            prioritize_high_end: Whether to prioritize high-end GPUs
            max_configs_per_model: Maximum number of GPU configs per model variant
            include_engines: List of inference engines to include (None = all)
            
        Returns:
            List of benchmark configurations sorted by priority
        """
        if include_engines is None:
            include_engines = list(self.inference_engines.keys())
        
        benchmark_plan = []
        
        for family, variants in self.model_mappings.items():
            for variant_size, model_info in variants.items():
                logger.info(f"Planning benchmarks for {family}-{variant_size}")
                
                # Get optimal GPUs for this model
                optimal_gpus = self.gpu_selector.select_optimal_gpus(
                    family, variant_size, count=max_configs_per_model
                )
                
                if not optimal_gpus:
                    logger.warning(f"No suitable GPUs found for {family}-{variant_size}")
                    continue
                
                for gpu_rank, gpu in enumerate(optimal_gpus):
                    for engine in include_engines:
                        try:
                            # Get engine-specific configuration
                            config = self.gpu_selector.get_model_config(
                                family, variant_size, gpu['id'], engine
                            )
                            
                            # Calculate priority score
                            priority = self._calculate_execution_priority(
                                family, variant_size, gpu, engine, gpu_rank
                            )
                            
                            benchmark_config = {
                                'id': f"{family}-{variant_size}-{engine}-{gpu['display_name'].replace(' ', '_')}",
                                'model_family': family,
                                'parameter_size': variant_size,
                                'model_name': f"{family.title()}-{variant_size.upper()}",
                                'inference_engine': engine,
                                'gpu_info': gpu,
                                'gpu_rank': gpu_rank + 1,
                                'config': config,
                                'priority': priority,
                                'model_ids': model_info,
                                'estimated_duration_minutes': self._estimate_benchmark_duration(
                                    family, variant_size, engine
                                ),
                                'estimated_cost_usd': self._estimate_benchmark_cost(
                                    family, variant_size, gpu, engine
                                )
                            }
                            
                            benchmark_plan.append(benchmark_config)
                            
                        except Exception as e:
                            logger.error(f"Failed to create config for {family}-{variant_size} on {gpu['id']} with {engine}: {e}")
                            continue
        
        # Sort by priority (highest first)
        benchmark_plan.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"Generated benchmark plan with {len(benchmark_plan)} configurations")
        return benchmark_plan
    
    def execute_benchmark_plan(self, 
                             benchmark_plan: List[Dict],
                             start_from_index: int = 0,
                             max_benchmarks: Optional[int] = None) -> Dict:
        """
        Execute the benchmark plan with error handling and progress tracking.
        
        Args:
            benchmark_plan: List of benchmark configurations to execute
            start_from_index: Index to start execution from (for resuming)
            max_benchmarks: Maximum number of benchmarks to run (None = all)
            
        Returns:
            Execution summary with results and statistics
        """
        self.execution_summary['start_time'] = time.time()
        self.execution_summary['total_benchmarks'] = len(benchmark_plan)
        
        if max_benchmarks:
            benchmark_plan = benchmark_plan[:max_benchmarks]
        
        if start_from_index > 0:
            benchmark_plan = benchmark_plan[start_from_index:]
            logger.info(f"Resuming execution from index {start_from_index}")
        
        logger.info(f"Starting execution of {len(benchmark_plan)} benchmarks")
        
        for i, config in enumerate(benchmark_plan, start_from_index):
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Executing benchmark {i+1}/{len(benchmark_plan) + start_from_index}")
                logger.info(f"Configuration: {config['id']}")
                logger.info(f"Model: {config['model_name']} ({config['parameter_size']})")
                logger.info(f"Engine: {config['inference_engine']}")
                logger.info(f"GPU: {config['gpu_info']['display_name']} ({config['gpu_info']['memory_gb']}GB)")
                logger.info(f"Estimated duration: {config['estimated_duration_minutes']} minutes")
                logger.info(f"Estimated cost: ${config['estimated_cost_usd']:.2f}")
                logger.info(f"{'='*80}")
                
                if self.dry_run:
                    logger.info("DRY RUN: Skipping actual execution")
                    self.execution_summary['skipped_benchmarks'] += 1
                    continue
                
                # Execute the benchmark
                result = self._execute_single_benchmark(config)
                
                if result['success']:
                    self.execution_summary['successful_benchmarks'] += 1
                    logger.info(f"✅ Benchmark completed successfully")
                else:
                    self.execution_summary['failed_benchmarks'] += 1
                    logger.error(f"❌ Benchmark failed: {result['error']}")
                
                # Store result
                result['config'] = config
                result['execution_index'] = i
                self.execution_summary['results'].append(result)
                
                # Save progress
                self._save_execution_progress()
                
                # Brief pause between benchmarks
                if i < len(benchmark_plan) - 1:
                    logger.info("Waiting 30 seconds before next benchmark...")
                    time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Benchmark execution interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in benchmark {i+1}: {e}")
                self.execution_summary['failed_benchmarks'] += 1
                
                error_result = {
                    'success': False,
                    'error': str(e),
                    'config': config,
                    'execution_index': i,
                    'timestamp': time.time()
                }
                self.execution_summary['results'].append(error_result)
                continue
        
        self.execution_summary['end_time'] = time.time()
        
        # Generate final summary
        self._generate_execution_summary()
        
        return self.execution_summary
    
    def _execute_single_benchmark(self, config: Dict) -> Dict:
        """Execute a single benchmark configuration."""
        start_time = time.time()
        
        try:
            engine = config['inference_engine']
            engine_info = self.inference_engines[engine]
            
            # Prepare arguments for the benchmark function
            benchmark_args = self._prepare_benchmark_args(config)
            
            # Execute the benchmark based on inference engine
            if engine == 'ollama':
                create_ollama_pod(**benchmark_args)
            elif engine == 'vllm':
                create_vllm_pod(**benchmark_args)
            elif engine == 'sglang':
                create_sglang_pod(**benchmark_args)
            else:
                raise ValueError(f"Unknown inference engine: {engine}")
            
            return {
                'success': True,
                'duration_seconds': time.time() - start_time,
                'timestamp': time.time(),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration_seconds': time.time() - start_time,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def _prepare_benchmark_args(self, config: Dict) -> Dict:
        """Prepare arguments for benchmark function calls."""
        engine = config['inference_engine']
        model_family = config['model_family']
        param_size = config['parameter_size']
        
        # Get the correct model ID based on inference engine
        if engine == 'ollama':
            model_id = config['model_ids']['ollama_id']
        else:  # vllm and sglang use HuggingFace model IDs
            model_id = config['model_ids']['hf_id']
        
        # Base arguments
        args = {
            'gpu_id': config['gpu_info']['id'],
            'volume_in_gb': config['config']['volume_in_gb'],
            'container_disk_in_gb': config['config']['container_disk_in_gb'],
            'llm_id': model_id,
            'port': config['config']['port'],
            'llm_parameter_size': param_size,
            'llm_common_name': config['model_name'],
            'gpu_count': 1  # Default to single GPU
        }
        
        # Adjust GPU count for large models if beneficial
        if param_size in ['405b', '72b', '180b']:
            # Only use multiple GPUs for extremely large models if available
            memory_required = config['config']['estimated_memory_usage']
            single_gpu_memory = config['gpu_info']['memory_gb']
            
            if memory_required > single_gpu_memory * 0.8:  # If close to memory limit
                args['gpu_count'] = min(2, int(memory_required / (single_gpu_memory * 0.7)))
        
        return args
    
    def _calculate_execution_priority(self, family: str, variant: str, gpu: Dict, engine: str, gpu_rank: int) -> float:
        """Calculate execution priority for benchmark ordering."""
        # Model family priorities
        family_scores = {'llama': 4.0, 'qwen': 3.0, 'gemma': 2.0, 'falcon': 1.0}
        
        # Inference engine priorities
        engine_scores = {'vllm': 3.0, 'sglang': 2.0, 'ollama': 1.0}
        
        # Parameter size priority (larger models = higher priority)
        param_count = float(variant.rstrip('b'))
        size_score = min(5.0, param_count / 50.0)  # Normalize to 0-5 scale
        
        # GPU rank penalty (prefer best GPU for each model)
        gpu_rank_penalty = gpu_rank * 0.5
        
        # GPU tier bonus
        gpu_tier_bonus = 0
        gpu_id_upper = gpu['id'].upper()
        if any(x in gpu_id_upper for x in ['H200', 'B200']):
            gpu_tier_bonus = 2.0
        elif any(x in gpu_id_upper for x in ['H100', 'A100']):
            gpu_tier_bonus = 1.5
        elif any(x in gpu_id_upper for x in ['L40S', 'RTX 6000']):
            gpu_tier_bonus = 1.0
        
        # Calculate final priority
        priority = (
            family_scores.get(family, 1.0) * 0.25 +
            engine_scores.get(engine, 1.0) * 0.25 +
            size_score * 0.25 +
            gpu_tier_bonus * 0.15 +
            10.0 * 0.1  # Base score
        ) - gpu_rank_penalty
        
        return max(0.1, priority)  # Ensure positive priority
    
    def _estimate_benchmark_duration(self, family: str, variant: str, engine: str) -> float:
        """Estimate benchmark duration in minutes."""
        param_count = float(variant.rstrip('b'))
        
        # Base duration factors
        base_duration = 15  # Base 15 minutes
        
        # Model size factor
        size_factor = 1.0 + (param_count / 100.0)  # Larger models take longer
        
        # Engine factor
        engine_factors = {'ollama': 1.2, 'vllm': 1.0, 'sglang': 1.1}
        engine_factor = engine_factors.get(engine, 1.0)
        
        # Family factor (some models are more complex)
        family_factors = {'llama': 1.0, 'qwen': 1.1, 'falcon': 1.2, 'gemma': 0.9}
        family_factor = family_factors.get(family, 1.0)
        
        total_duration = base_duration * size_factor * engine_factor * family_factor
        
        return round(total_duration, 1)
    
    def _estimate_benchmark_cost(self, family: str, variant: str, gpu: Dict, engine: str) -> float:
        """Estimate benchmark cost in USD."""
        duration_hours = self._estimate_benchmark_duration(family, variant, engine) / 60.0
        
        # Rough GPU hourly costs (these are estimates)
        gpu_costs_per_hour = {
            'H200': 4.0, 'H100': 3.0, 'A100': 2.5, 'B200': 5.0,
            'L40S': 1.5, 'L40': 1.3, 'A40': 1.2,
            'RTX 6000': 1.0, 'RTX 4090': 0.8, 'RTX 5090': 1.0
        }
        
        # Find cost for this GPU
        cost_per_hour = 1.0  # Default cost
        for gpu_type, cost in gpu_costs_per_hour.items():
            if gpu_type in gpu['id']:
                cost_per_hour = cost
                break
        
        return duration_hours * cost_per_hour
    
    def _save_execution_progress(self):
        """Save execution progress to file."""
        progress_file = Path("benchmark_execution_progress.json")
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(self.execution_summary, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _generate_execution_summary(self):
        """Generate and display execution summary."""
        total_duration = self.execution_summary['end_time'] - self.execution_summary['start_time']
        
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK EXECUTION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total benchmarks: {self.execution_summary['total_benchmarks']}")
        logger.info(f"Successful: {self.execution_summary['successful_benchmarks']}")
        logger.info(f"Failed: {self.execution_summary['failed_benchmarks']}")
        logger.info(f"Skipped: {self.execution_summary['skipped_benchmarks']}")
        logger.info(f"Success rate: {self.execution_summary['successful_benchmarks'] / max(1, self.execution_summary['total_benchmarks']) * 100:.1f}%")
        logger.info(f"Total duration: {total_duration / 3600:.2f} hours")
        logger.info(f"{'='*80}")
        
        # Save detailed summary
        summary_file = Path(f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(self.execution_summary, f, indent=2)
            logger.info(f"Detailed summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")


def main():
    """Main entry point for master benchmark runner."""
    parser = argparse.ArgumentParser(description="Master Benchmark Runner for LLM Inference Engines")
    
    parser.add_argument("--config", default="configs/runpod_gpus.yaml", 
                       help="Path to GPU configuration file")
    parser.add_argument("--mongo-url", help="MongoDB connection URL (overrides env var)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Generate plan without executing benchmarks")
    parser.add_argument("--max-benchmarks", type=int, 
                       help="Maximum number of benchmarks to run")
    parser.add_argument("--start-from", type=int, default=0, 
                       help="Start execution from specific index (for resuming)")
    parser.add_argument("--engines", nargs="+", choices=["ollama", "vllm", "sglang"],
                       help="Inference engines to include (default: all)")
    parser.add_argument("--max-configs-per-model", type=int, default=2,
                       help="Maximum GPU configurations to test per model variant")
    parser.add_argument("--plan-only", action="store_true",
                       help="Generate and display plan only (don't execute)")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = MasterBenchmarkRunner(
        config_path=args.config,
        mongo_url=args.mongo_url,
        dry_run=args.dry_run
    )
    
    # Generate benchmark plan
    logger.info("Generating benchmark plan...")
    benchmark_plan = runner.generate_benchmark_plan(
        max_configs_per_model=args.max_configs_per_model,
        include_engines=args.engines
    )
    
    # Display plan
    logger.info(f"\nGenerated {len(benchmark_plan)} benchmark configurations:")
    total_estimated_cost = 0
    total_estimated_duration = 0
    
    for i, config in enumerate(benchmark_plan):
        logger.info(f"{i+1:3d}. {config['id']} "
                   f"(Priority: {config['priority']:.2f}, "
                   f"Duration: {config['estimated_duration_minutes']}min, "
                   f"Cost: ${config['estimated_cost_usd']:.2f})")
        total_estimated_cost += config['estimated_cost_usd']
        total_estimated_duration += config['estimated_duration_minutes']
    
    logger.info(f"\nTotal estimated duration: {total_estimated_duration/60:.1f} hours")
    logger.info(f"Total estimated cost: ${total_estimated_cost:.2f}")
    
    if args.plan_only:
        # Save plan to file
        plan_file = Path(f"benchmark_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(plan_file, 'w') as f:
            json.dump(benchmark_plan, f, indent=2)
        logger.info(f"Benchmark plan saved to: {plan_file}")
        return
    
    # Execute benchmarks
    if not args.dry_run:
        confirmation = input(f"\nProceed with execution of {len(benchmark_plan)} benchmarks? (y/N): ")
        if confirmation.lower() != 'y':
            logger.info("Execution cancelled by user")
            return
    
    logger.info("\nStarting benchmark execution...")
    summary = runner.execute_benchmark_plan(
        benchmark_plan=benchmark_plan,
        start_from_index=args.start_from,
        max_benchmarks=args.max_benchmarks
    )
    
    logger.info("Benchmark execution completed!")


if __name__ == "__main__":
    main() 