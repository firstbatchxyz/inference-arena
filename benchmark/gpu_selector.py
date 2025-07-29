"""
GPU Selection Logic for Benchmark Optimization

This module provides intelligent GPU selection based on model requirements,
GPU specifications, and availability considerations.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

class GPUSelector:
    def __init__(self, gpu_config_path: str = "configs/runpod_gpus.yaml"):
        """Initialize GPU selector with GPU inventory."""
        self.gpu_config_path = Path(gpu_config_path)
        self.gpus = self._load_gpu_inventory()
        
    def _load_gpu_inventory(self) -> List[Dict]:
        """Load GPU inventory from YAML configuration."""
        with open(self.gpu_config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('gpus', [])
    
    def _estimate_model_memory_requirements(self, model_family: str, parameter_size: str) -> Dict[str, float]:
        """
        Estimate memory requirements for different model families and sizes.
        Returns dict with model_memory_gb, kv_cache_gb, and total_required_gb.
        """
        # Extract numeric parameter count
        param_match = re.search(r'(\d+(?:\.\d+)?)', parameter_size.lower())
        param_count = float(param_match.group(1)) if param_match else 1.0
        
        # Base memory calculations (rough estimates)
        if 'b' in parameter_size.lower():  # Billion parameters
            # Approximate: 2 bytes per parameter (bfloat16) + overhead
            model_memory_gb = param_count * 2.2  # 2 bytes + 10% overhead
        else:
            model_memory_gb = param_count * 0.002  # Fallback for smaller models
        
        # KV cache estimation (context-dependent)
        if param_count >= 70:
            kv_cache_gb = param_count * 0.5  # Large models need more KV cache
        elif param_count >= 30:
            kv_cache_gb = param_count * 0.3
        else:
            kv_cache_gb = param_count * 0.2
        
        # Model family specific adjustments
        family_multipliers = {
            'llama': 1.0,      # Baseline
            'qwen': 1.1,       # Slightly more memory for Qwen
            'falcon': 1.2,     # Falcon models are memory-heavy
            'gemma': 0.9       # Gemma models are more efficient
        }
        
        multiplier = family_multipliers.get(model_family.lower(), 1.0)
        model_memory_gb *= multiplier
        kv_cache_gb *= multiplier
        
        # Total required memory (includes OS, framework overhead)
        total_required_gb = (model_memory_gb + kv_cache_gb) * 1.25  # 25% safety margin
        
        return {
            'model_memory_gb': model_memory_gb,
            'kv_cache_gb': kv_cache_gb,
            'total_required_gb': total_required_gb
        }
    
    def _rank_gpu_suitability(self, gpu: Dict, memory_req: Dict, model_family: str) -> float:
        """
        Rank GPU suitability based on memory capacity, performance, and cost efficiency.
        Returns a score from 0-100 (higher is better).
        """
        memory_gb = gpu['memory_gb']
        required_gb = memory_req['total_required_gb']
        
        # Base score: memory adequacy
        if memory_gb < required_gb:
            return 0.0  # Insufficient memory
        
        memory_ratio = memory_gb / required_gb
        
        # Score components
        scores = {
            'memory_adequacy': min(100, (memory_ratio - 1) * 50 + 50),  # 50-100 points
            'gpu_tier': 0,
            'efficiency': 0
        }
        
        # GPU tier scoring (performance-based)
        gpu_id = gpu['id'].upper()
        if any(x in gpu_id for x in ['H200']):
            scores['gpu_tier'] = 100  # Latest generation (compatible)
        elif any(x in gpu_id for x in ['H100']):
            scores['gpu_tier'] = 95   # High-end current gen
        elif any(x in gpu_id for x in ['A100']):
            scores['gpu_tier'] = 90   # High-end previous gen
        elif any(x in gpu_id for x in ['B200']):
            # B200 compatibility penalty for vLLM (PyTorch sm_100 issue)
            scores['gpu_tier'] = 85 if model_family.lower() in ['qwen', 'llama'] else 75
        elif any(x in gpu_id for x in ['L40S', 'RTX 6000']):
            scores['gpu_tier'] = 85   # Workstation high-end
        elif any(x in gpu_id for x in ['A40', 'L40']):
            scores['gpu_tier'] = 80   # Datacenter mid-range
        elif any(x in gpu_id for x in ['RTX 4090', 'RTX 5090']):
            scores['gpu_tier'] = 75   # Consumer high-end
        elif any(x in gpu_id for x in ['RTX 4080', 'RTX 5080']):
            scores['gpu_tier'] = 70   # Consumer mid-high
        else:
            scores['gpu_tier'] = 50   # Other GPUs
        
        # Efficiency bonus for not over-provisioning
        if memory_ratio <= 2.0:  # Not wasting too much memory
            scores['efficiency'] = 20
        elif memory_ratio <= 3.0:
            scores['efficiency'] = 10
        else:
            scores['efficiency'] = 0
        
        # Model family specific preferences
        family_preferences = {
            'llama': {'H100': 5, 'A100': 5, 'H200': 10},
            'qwen': {'H200': 10, 'H100': 5, 'A100': 3},
            'falcon': {'H200': 10, 'H100': 8, 'A100': 5},
            'gemma': {'RTX 4090': 5, 'L4': 3, 'A100': 8}
        }
        
        family_bonus = 0
        if model_family.lower() in family_preferences:
            for gpu_type, bonus in family_preferences[model_family.lower()].items():
                if gpu_type in gpu_id:
                    family_bonus = bonus
                    break
        
        # Final weighted score
        final_score = (
            scores['memory_adequacy'] * 0.4 +
            scores['gpu_tier'] * 0.4 +
            scores['efficiency'] * 0.15 +
            family_bonus * 0.05
        )
        
        return final_score
    
    def select_optimal_gpus(self, model_family: str, parameter_size: str, count: int = 3) -> List[Dict]:
        """
        Select the most suitable GPUs for a given model family and parameter size.
        
        Args:
            model_family: Model family (llama, qwen, falcon, gemma)
            parameter_size: Parameter size (e.g., "7b", "70b", "405b")
            count: Number of GPU options to return
            
        Returns:
            List of GPU configurations sorted by suitability (best first)
        """
        memory_requirements = self._estimate_model_memory_requirements(model_family, parameter_size)
        
        # Score all GPUs
        gpu_scores = []
        for gpu in self.gpus:
            score = self._rank_gpu_suitability(gpu, memory_requirements, model_family)
            if score > 0:  # Only include GPUs with sufficient memory
                gpu_scores.append({
                    'gpu': gpu,
                    'score': score,
                    'memory_req': memory_requirements
                })
        
        # Sort by score (descending) and return top choices
        gpu_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return [item['gpu'] for item in gpu_scores[:count]]
    
    def get_model_config(self, model_family: str, parameter_size: str, gpu_id: str, inference_engine: str) -> Dict:
        """
        Get optimized configuration for a specific model, GPU, and inference engine combination.
        """
        memory_req = self._estimate_model_memory_requirements(model_family, parameter_size)
        
        # Find the GPU info
        gpu_info = next((gpu for gpu in self.gpus if gpu['id'] == gpu_id), None)
        if not gpu_info:
            raise ValueError(f"GPU {gpu_id} not found in inventory")
        
        # Base configuration
        config = {
            'gpu_id': gpu_id,
            'gpu_memory_gb': gpu_info['memory_gb'],
            'model_family': model_family,
            'parameter_size': parameter_size,
            'estimated_memory_usage': memory_req['total_required_gb'],
            'memory_utilization_ratio': memory_req['total_required_gb'] / gpu_info['memory_gb']
        }
        
        # Inference engine specific configurations
        if inference_engine.lower() == 'vllm':
            config.update(self._get_vllm_config(model_family, parameter_size, gpu_info))
        elif inference_engine.lower() == 'sglang':
            config.update(self._get_sglang_config(model_family, parameter_size, gpu_info))
        elif inference_engine.lower() == 'ollama':
            config.update(self._get_ollama_config(model_family, parameter_size, gpu_info))
        
        return config
    
    def _get_vllm_config(self, model_family: str, parameter_size: str, gpu_info: Dict) -> Dict:
        """Get vLLM-specific configuration."""
        param_count = float(re.search(r'(\d+(?:\.\d+)?)', parameter_size).group(1))
        
        config = {
            'volume_in_gb': max(100, int(param_count * 3)),  # Model size + cache
            'container_disk_in_gb': max(200, int(param_count * 5)),
            'port': 8000
        }
        
        # Adjust for large models
        if param_count >= 70:
            config['volume_in_gb'] = max(config['volume_in_gb'], 400)
            config['container_disk_in_gb'] = max(config['container_disk_in_gb'], 800)
        
        return config
    
    def _get_sglang_config(self, model_family: str, parameter_size: str, gpu_info: Dict) -> Dict:
        """Get SGLang-specific configuration."""
        param_count = float(re.search(r'(\d+(?:\.\d+)?)', parameter_size).group(1))
        
        config = {
            'volume_in_gb': max(100, int(param_count * 3)),
            'container_disk_in_gb': max(200, int(param_count * 5)),
            'port': 30000
        }
        
        if param_count >= 70:
            config['volume_in_gb'] = max(config['volume_in_gb'], 300)
            config['container_disk_in_gb'] = max(config['container_disk_in_gb'], 600)
        
        return config
    
    def _get_ollama_config(self, model_family: str, parameter_size: str, gpu_info: Dict) -> Dict:
        """Get Ollama-specific configuration."""
        param_count = float(re.search(r'(\d+(?:\.\d+)?)', parameter_size).group(1))
        
        config = {
            'volume_in_gb': max(80, int(param_count * 2)),
            'container_disk_in_gb': max(150, int(param_count * 3)),
            'port': 11434
        }
        
        if param_count >= 70:
            config['volume_in_gb'] = max(config['volume_in_gb'], 200)
            config['container_disk_in_gb'] = max(config['container_disk_in_gb'], 400)
        
        return config
    
    def generate_benchmark_matrix(self) -> List[Dict]:
        """
        Generate a complete benchmark matrix for all model families and variants.
        """
        # Define model families and their variants based on ollama_available_llms.yaml
        model_matrix = {
            'gemma': ['1b', '4b', '12b'],
            'falcon': ['10b', '180b'],
            'qwen': ['14b', '32b', '72b'],
            'llama': ['70b', '405b']
        }
        
        inference_engines = ['ollama', 'vllm', 'sglang']
        
        benchmark_configs = []
        
        for family, variants in model_matrix.items():
            for variant in variants:
                # Select top 3 GPUs for this model
                optimal_gpus = self.select_optimal_gpus(family, variant, count=3)
                
                for i, gpu in enumerate(optimal_gpus):
                    for engine in inference_engines:
                        try:
                            config = self.get_model_config(family, variant, gpu['id'], engine)
                            
                            benchmark_configs.append({
                                'model_family': family,
                                'parameter_size': variant,
                                'gpu_rank': i + 1,  # 1st, 2nd, 3rd choice
                                'inference_engine': engine,
                                'config': config,
                                'priority': self._calculate_priority(family, variant, gpu, engine)
                            })
                        except Exception as e:
                            print(f"Warning: Could not generate config for {family}-{variant} on {gpu['id']} with {engine}: {e}")
                            continue
        
        # Sort by priority (higher priority first)
        benchmark_configs.sort(key=lambda x: x['priority'], reverse=True)
        
        return benchmark_configs
    
    def _calculate_priority(self, family: str, variant: str, gpu: Dict, engine: str) -> float:
        """Calculate benchmark priority score."""
        # Base priority factors
        family_priority = {'llama': 4, 'qwen': 3, 'gemma': 2, 'falcon': 1}
        engine_priority = {'vllm': 3, 'sglang': 2, 'ollama': 1}
        
        param_count = float(re.search(r'(\d+(?:\.\d+)?)', variant).group(1))
        size_priority = min(5, param_count / 20)  # Larger models get higher priority
        
        gpu_tier_priority = 3 if any(x in gpu['id'].upper() for x in ['H200', 'H100', 'A100']) else 1
        
        return (
            family_priority.get(family, 1) * 0.3 +
            engine_priority.get(engine, 1) * 0.2 +
            size_priority * 0.3 +
            gpu_tier_priority * 0.2
        ) 