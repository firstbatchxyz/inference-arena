import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)

class RunpodClient:
    """
    Simple GraphQL client for RunPod with connection pooling and retry logic.
    """

    def __init__(self, api_key=None):
        # GraphQL endpoint and auth header
        self.endpoint = "https://api.runpod.io/graphql"
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Create session with connection pooling and retry strategy
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(self.headers)

    def _post(self, query: str, variables: dict = None) -> dict:
        """Internal: send a GraphQL POST and return the 'data' field."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        try:
            resp = self.session.post(self.endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            
            json_data = resp.json()
            if "errors" in json_data:
                logger.error(f"GraphQL errors: {json_data['errors']}")
                raise ValueError(f"GraphQL errors: {json_data['errors']}")
            
            return json_data.get("data", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"RunPod API request failed: {e}")
            raise


    def get_pod_info(self, pod_id: str) -> dict:
        """
        Fetch basic pod info including costPerHr and runtime.uptimeInSeconds.
        """
        query = """
        query GetPod($input: PodFilter!) {
          pod(input: $input) {
            id name costPerHr uptimeSeconds
            runtime { uptimeInSeconds }
          }
        }"""
        return self._post(query, {"input": {"podId": pod_id}})["pod"]  # :contentReference[oaicite:6]{index=6}

    def calculate_used_balance(self, pod_id: str) -> dict:
        """
        Calculate the used balance: (uptimeInSeconds / 3600) * costPerHr.
        """
        pod = self.get_pod_info(pod_id)
        
        # Prefer runtime.uptimeInSeconds if available; fallback to uptimeSeconds
        uptime = pod.get("runtime", {}).get("uptimeInSeconds") or pod.get("uptimeSeconds", 0)
        hours = uptime / 3600  # seconds → hours :contentReference[oaicite:7]{index=7}
        cost = float(pod["costPerHr"])
        used_balance = hours * cost
        return {
            "pod_id": pod_id,
            "uptime_seconds": uptime,
            "cost_per_hr": cost,
            "used_balance": used_balance
        }
    
    def close(self):
        """Close the session to clean up connections."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
