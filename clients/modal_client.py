"""
Modal client for cost calculation.
Since Modal doesn't expose a billing API, we calculate costs manually based on GPU pricing.
"""


# Modal GPU pricing per second (as of 2024)
# Source: https://modal.com/pricing
MODAL_GPU_PRICING_PER_SEC = {
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,
    "A100": 0.000583,  # Defaults to 40GB in Modal
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "L40S": 0.000542,
    "A10": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}


class ModalClient:
    """
    Client for calculating Modal costs.
    Since Modal doesn't expose a billing API, we calculate costs manually
    based on GPU pricing and runtime.
    """

    def __init__(self):
        """Initialize Modal client with GPU pricing."""
        self.pricing = MODAL_GPU_PRICING_PER_SEC

    def calculate_cost(
        self, gpu_id: str, gpu_count: int, runtime_seconds: float
    ) -> dict:
        """
        Calculate Modal cost based on runtime and GPU pricing.
        
        Args:
            gpu_id: GPU type (e.g., "H100", "A100", "L4")
            gpu_count: Number of GPUs
            runtime_seconds: Total runtime in seconds
        
        Returns:
            dict with cost information (minimal, non-redundant fields):
            {
                "runtime_seconds": float,
                "used_balance": float,
            }
        """
        # Normalize GPU ID (handle variations like "A100-80GB" or just "A100")
        gpu_key = gpu_id.upper()
        
        # Try exact match first
        if gpu_key not in self.pricing:
            # Try without suffix (e.g., "A100-80GB" -> "A100")
            base_gpu = gpu_key.split("-")[0]
            if base_gpu in self.pricing:
                gpu_key = base_gpu
            else:
                # Unknown GPU type, return error info
                return {
                    "error": f"Unknown GPU type: {gpu_id}. Available: {list(self.pricing.keys())}",
                    "runtime_seconds": runtime_seconds,
                    "used_balance": 0.0,
                }
        
        cost_per_sec = self.pricing[gpu_key]
        total_cost = cost_per_sec * gpu_count * runtime_seconds
        
        return {
            "runtime_seconds": runtime_seconds,
            "used_balance": total_cost,
        }

