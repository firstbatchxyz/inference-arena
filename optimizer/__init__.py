"""High-level interface for the Inference Arena optimizer."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
_LLM_OPTIMIZER_SRC = _ROOT / "llm-optimizer" / "src"
if _LLM_OPTIMIZER_SRC.exists():
    src_path = str(_LLM_OPTIMIZER_SRC)
    if src_path not in sys.path:
        sys.path.append(src_path)

from .advisor import LLMGPUAdvisor
from .models import (
    GPURecommendation,
    LLMGPUAnalysis,
    OptimalConfig,
    SystemCompatibility,
)
from .reporting import format_analysis

__all__ = [
    "LLMGPUAdvisor",
    "GPURecommendation",
    "SystemCompatibility",
    "OptimalConfig",
    "LLMGPUAnalysis",
    "format_analysis",
]
