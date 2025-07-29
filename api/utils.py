import os
import json
from benchmark.mongo_client import Mongo
from .ai_client import BenchmarkAIClient
import math
from typing import Optional
from .models import PaginationMeta

def build_benchmark_query(
        model_name=None,
        inference_engine=None,
        parameter_size=None,
        hardware=None,
        hardware_price=None,
        TTFT_max=None,
        TTFT_min=None,
        TPS_max=None,
        TPS_min=None,
        benchmark_type=None,
        hourly_cost_min=None,
        hourly_cost_max=None
    ):
        query = {}
        if model_name:
            query["llm_common_name"] = model_name
        if inference_engine:
            query["inference_name"] = inference_engine
        if parameter_size:
            query["llm_parameter_size"] = parameter_size
        if hardware:
            if isinstance(hardware, list) and len(hardware) > 0:
                if len(hardware) == 1:
                    query["gpu_id"] = hardware[0]
                else:
                    query["gpu_id"] = {"$in": hardware}
            elif isinstance(hardware, str):
                query["gpu_id"] = hardware
        if hardware_price:
            query["pod_cost.cost_per_hr"] = hardware_price
        if TTFT_max:
            query["time_to_first_token_ms"] = {"$lte": TTFT_max}
        if TTFT_min:
            query["time_to_first_token_ms"] = {"$gte": TTFT_min}
        if TTFT_max and TTFT_min:
            query["time_to_first_token_ms"] = {"$gte": TTFT_min, "$lte": TTFT_max}
        if TPS_max:
            query["output_tokens_per_second"] = {"$lte": TPS_max}
        if TPS_min:
            query["output_tokens_per_second"] = {"$gte": TPS_min}
        if TTFT_max and TTFT_min:
            query["time_to_first_token_ms"] = {"$gte": TTFT_min, "$lte": TTFT_max}
        if TPS_max and TPS_min:
            query["output_tokens_per_second"] = {"$gte": TPS_min, "$lte": TPS_max}
        if benchmark_type:
            query["benchmark_type"] = benchmark_type
        if hourly_cost_min:
            query["pod_cost.cost_per_hr"] = {"$gte": hourly_cost_min}
        if hourly_cost_max:
            query["pod_cost.cost_per_hr"] = {"$lte": hourly_cost_max}
        if hourly_cost_min and hourly_cost_max:
            query["pod_cost.cost_per_hr"] = {"$gte": hourly_cost_min, "$lte": hourly_cost_max}
        return query

def get_gpu_prices_by_pod_id_util(pod_id: str):
    """
    Utility function to get GPU prices by pod ID
    Returns a list of dictionaries with GPU pricing information
    Raises ValueError for missing pod or GPU information
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    pod_benchmark = mongo_client.find_one("pod_benchmarks", {"pod_id": pod_id})
    
    if not pod_benchmark:
        raise ValueError("Pod not found")

    pod_gpu = pod_benchmark.get("gpu_id")
    if not pod_gpu:
        raise ValueError("GPU information not found for this pod")
        
    all_gpu_prices = mongo_client.find_many("gpu_pricing_processed", {})
    
    ai_client = BenchmarkAIClient()
    
    gpu_prices_response = ai_client.chat(message=f"""
        
        You are given a GPU name: {pod_gpu}

        And a list of GPU prices with corresponding platforms: {list(all_gpu_prices)}

        Your task is:
        - Read and compare all entries in the provided list.
        - Identify the GPU(s) from the list that match or are equivalent to the given GPU name, even if the names are not exactly the same (e.g., slight differences or aliases).
        - Return a list of matching entries in the following format:

        [
        {{
            "name": "<exact name from the list that matched>",
            "price": <price>,
            "platform": "<platform>"
        }},
        ...
        ]

        Important rules:
        - Your output must only be a list of dictionaries.
        - Do not return any explanations, comments, or extra text.
        - Ensure the GPU name comparison is fuzzy or semantic-aware to catch similar names.
        - All GPU names should be same in returned response.

        Begin now.
        """)

    try:
        # Parse the JSON string response from AI client
        gpu_prices_list = json.loads(gpu_prices_response)

        # Ensure it's a list
        if not isinstance(gpu_prices_list, list):
            raise ValueError("AI response is not a list")
            
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback: try to extract JSON from the response if it contains extra text
        try:
            # Look for JSON array in the response
            start_idx = gpu_prices_response.find('[')
            end_idx = gpu_prices_response.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = gpu_prices_response[start_idx:end_idx]
                gpu_prices_list = json.loads(json_str)
            else:
                raise ValueError(f"Failed to parse AI response: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse AI response: {str(e)}")

    for gpu_price in gpu_prices_list:
        gpu_price["platform"] = gpu_price["platform"].capitalize()
        if gpu_price.get("platform") == "Primeintellect":
            gpu_price["platform"] = "Prime Intellect"
        elif gpu_price.get("platform") == "Akash":
            gpu_price["platform"] = "Akash Network"
    
    return gpu_prices_list

def create_pagination_meta(current_page: int, per_page: int, total_items: int) -> PaginationMeta:
    """
    Create pagination metadata
    """
    total_pages = math.ceil(total_items / per_page) if total_items > 0 else 1
    has_next = current_page < total_pages
    has_prev = current_page > 1
    
    return PaginationMeta(
        current_page=current_page,
        per_page=per_page,
        total_items=total_items,
        total_pages=total_pages,
        has_next=has_next,
        has_prev=has_prev
    )
