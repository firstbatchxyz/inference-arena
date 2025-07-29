import os
import json
from typing import List, Dict, Any, Optional, AsyncIterator
from openai import OpenAI
from dotenv import load_dotenv
from benchmark.mongo_client import Mongo
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class BenchmarkQuery(BaseModel):
    """Query model for searching benchmark data"""
    model_name: Optional[str] = Field(None, description="Model name to search for")
    inference_engine: Optional[str] = Field(None, description="Inference engine (Ollama, VLLM, etc.)")
    gpu_type: Optional[str] = Field(None, description="GPU type to filter by")
    parameter_size: Optional[str] = Field(None, description="Model parameter size")
    benchmark_type: Optional[str] = Field(None, description="Type of benchmark (throughput, synchronous, etc.)")
    min_throughput: Optional[float] = Field(None, description="Minimum throughput requirement")
    max_latency: Optional[float] = Field(None, description="Maximum latency requirement")
    sort_by: Optional[str] = Field(None, description="Field to sort results by")
    limit: Optional[int] = Field(10, description="Maximum number of results to return")


class BenchmarkAIClient:
    """AI Client with OpenAI integration and MongoDB search capabilities for benchmark data analysis"""
    
    def __init__(self, openai_api_key: str = None, mongo_url: str = None):
        """
        Initialize the AI client with OpenAI and MongoDB connections
        
        Args:
            openai_api_key: OpenAI API key (defaults to env var OPENAI_API_KEY)
            mongo_url: MongoDB connection URL (defaults to env var MONGODB_URL)
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.mongo_url = mongo_url or os.getenv("MONGODB_URL")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.mongo_client = Mongo(self.mongo_url) if self.mongo_url else None
        
        # Define available tools for function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_benchmarks",
                    "description": "Search for benchmark results in MongoDB based on various criteria",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Model name to search for (e.g., 'llama2', 'mistral')"
                            },
                            "inference_engine": {
                                "type": "string",
                                "description": "Inference engine (e.g., 'Ollama', 'VLLM')"
                            },
                            "gpu_type": {
                                "type": "string",
                                "description": "GPU type (e.g., 'NVIDIA A100', 'NVIDIA H100')"
                            },
                            "parameter_size": {
                                "type": "string",
                                "description": "Model parameter size (e.g., '7B', '13B', '70B')"
                            },
                            "benchmark_type": {
                                "type": "string",
                                "description": "Type of benchmark (e.g., 'throughput', 'synchronous')"
                            },
                            "min_throughput": {
                                "type": "number",
                                "description": "Minimum output tokens per second"
                            },
                            "max_latency": {
                                "type": "number",
                                "description": "Maximum latency in milliseconds"
                            },
                            "sort_by": {
                                "type": "string",
                                "description": "Field to sort by (e.g., 'throughput', 'latency', 'cost')"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            }
                        }
                    }
                }
            },
        ]
    
    def search_benchmarks(self, **kwargs) -> List[Dict[str, Any]]:
        """Search for benchmark results in MongoDB"""
        if not self.mongo_client:
            return [{"error": "MongoDB client not initialized"}]
        
        # Build MongoDB query
        query = {}
        
        # Map parameters to MongoDB fields
        if kwargs.get("model_name"):
            query["llm_common_name"] = {"$regex": kwargs["model_name"], "$options": "i"}
        if kwargs.get("inference_engine"):
            query["inference_name"] = kwargs["inference_engine"]
        if kwargs.get("gpu_type"):
            query["gpu_id"] = {"$regex": kwargs["gpu_type"], "$options": "i"}
        if kwargs.get("parameter_size"):
            query["llm_parameter_size"] = kwargs["parameter_size"]
        if kwargs.get("benchmark_type"):
            query["benchmark_type"] = kwargs["benchmark_type"]
        
        # Get results from pod_benchmarks collection
        pod_results = list(self.mongo_client.find_many("pod_benchmarks", query))
        
        # Get benchmark results for each pod
        results = []
        for pod in pod_results[:kwargs.get("limit", 10)]:
            benchmark_query = {"pod_id": pod["pod_id"]}
            if kwargs.get("benchmark_type"):
                benchmark_query["benchmark_type"] = kwargs["benchmark_type"]
            else:
                benchmark_query["benchmark_type"] = "throughput"
            
            benchmark_result = self.mongo_client.find_one("benchmark_results", benchmark_query)
            
            if benchmark_result:
                # Apply filters
                if kwargs.get("min_throughput"):
                    throughput = benchmark_result.get("output_tokens_per_second", {}).get("total", {}).get("mean", 0)
                    if throughput < kwargs["min_throughput"]:
                        continue
                
                if kwargs.get("max_latency"):
                    latency = benchmark_result.get("request_latency", {}).get("total", {}).get("mean", float('inf'))
                    if latency > kwargs["max_latency"]:
                        continue
                
                # Combine pod and benchmark data
                result = {
                    "model_name": pod.get("llm_id"),
                    "parameter_size": pod.get("llm_parameter_size"),
                    "inference_engine": pod.get("inference_name"),
                    "gpu_type": pod.get("gpu_id"),
                    "server_type": pod.get("server_type"),
                    "pod_id": pod.get("pod_id"),
                    "cost_per_hour": pod.get("pod_cost", {}).get("cost_per_hr"),
                    "throughput": benchmark_result.get("output_tokens_per_second", {}),
                    "latency_ms": benchmark_result.get("request_latency", {}),
                    "ttft_ms": benchmark_result.get("time_to_first_token_ms", {}),
                    "total_requests": benchmark_result.get("total_requests"),
                    "successful_requests": benchmark_result.get("successful_requests"),
                    "benchmark_duration": benchmark_result.get("benchmark_duration")
                }
                results.append(result)
        
        # Sort results if requested
        if kwargs.get("sort_by"):
            sort_field = kwargs["sort_by"]
            if sort_field == "throughput":
                results.sort(key=lambda x: x.get("throughput", 0), reverse=True)
            elif sort_field == "latency":
                results.sort(key=lambda x: x.get("latency_ms", float('inf')))
            elif sort_field == "cost":
                results.sort(key=lambda x: x.get("cost_per_hour", float('inf')))
        
        return results
    
    
    def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool function based on its name"""
        if function_name == "search_benchmarks":
            return self.search_benchmarks(**arguments)
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def create_system_prompt(self, benchmark_data: Optional[str] = None,gpu_pricing_data: Optional[str] = None) -> str:
        """Create system prompt with benchmark data context"""
        base_prompt = f""" You are an AI assistant and a member of the Dria team, specializing in analyzing and answering questions about the Large Language Model (LLM) inference benchmark data.

        About Dria:
        Dria is the universal execution layer for AI, optimizing every model across all hardware. Dria operates a global network where anyone can contribute hardware to run large AI models. With a custom compiler for distributed inference, a decentralized resource-sharing network, and an evolving agent system for coding challenges, Dria aims to make advanced AI accessible, collaborative, and evolutionary. Currently, over 30,000 contributors serve inference through Dria's batch inference API (https://dria.co/batch-inference).

        You have access to a MongoDB database containing detailed benchmark results for various LLMs running on different hardware configurations and inference engines.

        The benchmark data includes:
        • Model performance metrics (e.g., throughput, latency, time to first token)
        • Hardware specifications (e.g., GPU types, memory)
        • Cost information (e.g., hourly rates, total costs)
        • Inference engine details (e.g., Ollama, VLLM)
        • Request statistics (e.g., total requests, success rates)

        You can search and analyze this data using the following tool:
        1. `search_benchmarks`: Search for benchmark results based on various criteria.

        When answering questions:
        • Always return your response in Markdown format only.
        • Use the appropriate tools to gather relevant data.
        • Provide specific metrics and comparisons when possible.
        • Consider cost-performance trade-offs.
        • Explain the significance of different metrics (e.g., throughput vs latency).
        • Try to understand the user's workload and use case when providing opinions or recommendations.
        • If you cannot find relevant benchmarks, ask the whether it wants to try with different configurations or suggest searching that benchmark manually then if fails they can request on-demand benchmarks for missing results.

        CAUTION Gently reject answering questions out of the platform's context which are inference, LLMs, AI infrastructure etc. by saying something like I'm sory but I can only help you with inference related questions..

        Current benchmark context which the user views on the current screen while chatting with you:
        {benchmark_data}

        Current GPU pricing context which the user views on the current screen while chatting with you:
        {gpu_pricing_data}
            """
        
        
        return base_prompt
    
    def chat(self, message: str, benchmark_data: Optional[str] = None, gpu_pricing_data: Optional[str] = None,
             conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Send a message to the AI and get a response
        
        Args:
            message: User's message/question
            benchmark_data: Optional benchmark data to include in system prompt
            conversation_history: Optional list of previous messages
            
        Returns:
            AI's response as a string
        """
        # Prepare messages
        messages = [
            {"role": "system", "content": self.create_system_prompt(benchmark_data,gpu_pricing_data)}
        ]
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Initial API call with tools
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message.model_dump())
        
        # Handle tool calls if any
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the function
                function_response = self._execute_function(function_name, function_args)
                
                # Add function response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
            
            # Get final response after tool execution
            second_response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages
            )
            
            return second_response.choices[0].message.content
        
        return response_message.content

    async def chat_stream(self, message: str, benchmark_data: Optional[str] = None, gpu_pricing_data: Optional[str] = None,
                         conversation_history: Optional[List[Dict[str, str]]] = None):
        """
        Send a message to the AI and get a streaming response
        
        Args:
            message: User's message/question
            benchmark_data: Optional benchmark data to include in system prompt
            gpu_pricing_data: Optional GPU pricing data to include in system prompt
            conversation_history: Optional list of previous messages
            
        Yields:
            Streaming chunks of AI's response
        """
        # Prepare messages
        messages = [
            {"role": "system", "content": self.create_system_prompt(benchmark_data, gpu_pricing_data)}
        ]
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # First check if we need to use tools (non-streaming for tool calls)
        initial_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            max_tokens=1  # Minimal response to check for tool calls
        )
        
        response_message = initial_response.choices[0].message
        
        # Handle tool calls if any (non-streaming)
        if response_message.tool_calls:
            messages.append(response_message.model_dump())
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the function
                function_response = self._execute_function(function_name, function_args)
                
                # Add function response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
            
            # Get streaming response after tool execution
            stream = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True
            )
        else:
            # Direct streaming response without tool calls
            stream = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True
            )
        
        # Stream the response
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


# Example usage and testing
if __name__ == "__main__":
    # Initialize the AI client
    ai_client = BenchmarkAIClient()
    
    # Example benchmark data to include in system prompt
    example_benchmark_data = """
    Recent benchmark results show:
    - Llama 2 7B on NVIDIA A100: 150 tokens/sec throughput, 45ms TTFT
    - Mistral 7B on NVIDIA H100: 250 tokens/sec throughput, 30ms TTFT
    - Cost ranges from $0.80/hr (A100) to $2.50/hr (H100)
    """
    
    # Example queries
    queries = [
        "What are the fastest models in the benchmark database?",
        "Compare llama2 and mistral performance",
        "Which GPU provides the best cost-performance ratio?",
        "Show me benchmarks for models running on NVIDIA H100",
        "What's the average latency for 7B parameter models?"
    ]
    
    
    for query in queries:
        response = ai_client.chat(query, benchmark_data=example_benchmark_data)



