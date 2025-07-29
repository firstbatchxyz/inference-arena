from fastapi import FastAPI, HTTPException, Query, Path, Depends, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from datetime import datetime
from typing import Optional, List
import uvicorn
import asyncio
import json as json_module
from benchmark.mongo_client import Mongo
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .models import APIResponse, BenchmarkResultInput, LoginResponse, User, CommentInput, CommentResponse, LikeInput, PodUpvoteInput, ChatMessageInput, ChatLogResponse, ChatLogDeleteInput, TokenUpdateRequest, TokenResponse, GithubLoginResponse, GithubCallbackResponse, AuthCurrentUserResponse, AuthLogoutResponse, AuthRefreshTokenResponse, BenchmarkResult, BenchmarkResultList, BenchmarkResultFilterInput, BenchmarkResultFilterResponse, ChatResponse, GetCommentsResponse, GetAllCommentsResponse, PostCommentResponse, DeleteCommentResponse, RemoveLikeResponse, LikeResponse,GetUserCommentsResponse,GetChatLogsResponse,GetUserChatLogsResponse,GetPodChatLogsResponse,DeleteChatLogResponse,GetGpuPricesByPlatformResponse,GpuPrice,GetAvailableBenchmarkOptionsResponse,ContentModerationError
from .auth import AuthService, get_current_user, get_current_user_optional
from .cache import cache_result, invalidate_benchmark_cache, invalidate_comments_cache, invalidate_gpu_cache
import os
from .ai_client import BenchmarkAIClient
from bson.objectid import ObjectId
from .utils import build_benchmark_query, get_gpu_prices_by_pod_id_util
from .logging_config import setup_logging, get_logger, LogContext
from .error_handlers import (
    APIError, SecurityError, ResourceNotFoundError, ValidationError as CustomValidationError,
    DatabaseError, ExternalServiceError, api_error_handler, validation_error_handler,
    rate_limit_handler, http_exception_handler, general_exception_handler
)
import json
# Setup structured logging
setup_logging()
logger = get_logger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Mock Dria Benchmark API",
    description="Mock Dria Benchmark API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiter to app state
app.state.limiter = limiter

# Add comprehensive error handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Add CORS middleware with secure configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "https://dria.co,https://app.dria.co,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

def calculate_ranking_score(likes: int, created_at: datetime, reply_count: int = 0) -> float:
    """
    Calculate ranking score based on likes, time, and engagement
    Uses Reddit-style ranking algorithm
    """
    # Time decay factor (comments get older)
    time_diff = datetime.utcnow() - created_at
    time_factor = max(1, time_diff.total_seconds() / 45000)  # 12.5 hours
    
    # Engagement factor (replies increase score)
    engagement_factor = 1 + (reply_count * 0.1)
    
    # Calculate score using Reddit's algorithm
    score = (likes * engagement_factor) / (time_factor ** 1.5)
    
    return round(score, 4)



# Root endpoint
@app.get("/", response_model=APIResponse)
async def root():
    """Welcome endpoint"""
    return APIResponse(
        success=True,
        message="Welcome to the FastAPI Example API!",
        data={"version": "1.0.0", "docs": "/docs"}
    )

# Health check endpoint
@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    return APIResponse(
        success=True,
        message="API is healthy",
        data={"timestamp": datetime.now().isoformat()}
    )

# Authentication endpoints
@app.get("/auth/github/login", response_model=GithubLoginResponse)
async def github_login():
    """Get GitHub OAuth login URL"""
    github_client_id = os.getenv("GITHUB_CLIENT_ID")
    redirect_uri = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback")
    
    if not github_client_id:
        raise HTTPException(
            status_code=500,
            detail="GitHub OAuth not configured. Please set GITHUB_CLIENT_ID environment variable."
        )
    
    auth_url = f"https://github.com/login/oauth/authorize?client_id={github_client_id}&redirect_uri={redirect_uri}&scope=user:email,read:user"
    
    return GithubLoginResponse(
        auth_url=auth_url
    )

@app.get("/auth/github/callback")
async def github_callback(code: str):
    """Handle GitHub OAuth callback and redirect to dria.co with token"""
    if not code:
        raise HTTPException(
            status_code=400,
            detail="Authorization code is required"
        )
    
    auth_service = AuthService()
    login_response = await auth_service.authenticate_github_user(code)
    
    # Redirect to dria.co with the access token as a query parameter
    redirect_url = f"https://dria.co/inference-benchmark?token={login_response.access_token}"
    
    return RedirectResponse(url=redirect_url, status_code=302)

@app.get("/auth/me", response_model=AuthCurrentUserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current authenticated user information"""
    return AuthCurrentUserResponse(
        user=current_user
    )

@app.get("/auth/logout", response_model=AuthLogoutResponse)
async def logout():
    """Logout endpoint (client should discard the token)"""
    return AuthLogoutResponse(
        success=True,
        message="Successfully logged out. Please discard your access token.",
    )

@app.post("/auth/refresh", response_model=AuthRefreshTokenResponse)
async def refresh_token(current_user: User = Depends(get_current_user)):
    """Refresh JWT access token"""
    auth_service = AuthService()
    
    # Verify user still exists in database
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    user_data = mongo_client.find_one("users", {"github_id": current_user.github_id})
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found in database"
        )
    
    # Update last login time
    mongo_client.update_one(
        "users",
        {"github_id": current_user.github_id},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    # Create new JWT token with same user data
    new_access_token = auth_service.create_access_token(
        data={"sub": str(current_user.github_id), "username": current_user.username}
    )
    
    return AuthRefreshTokenResponse(
        access_token=new_access_token,
        token_type="bearer",
        user=current_user
    )

@app.get("/benchmark_results", response_model=BenchmarkResult)
@cache_result(ttl_seconds=600, cache_prefix="benchmark")  # Cache for 10 minutes
async def benchmark_results(pod_id: Optional[str] = None, platform: Optional[str] = None, gpu_type: Optional[str] = None, llm_id: Optional[str] = None, server_type: Optional[str] = None, current_user: Optional[User] = Depends(get_current_user_optional)):
    
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    query = {}

    print(current_user)
    ## Find related pod log
    if pod_id:
        query["pod_id"] = pod_id
    if platform:
        query["inference_name"] = platform
    if gpu_type:
        query["gpu_id"] = gpu_type
    if llm_id:
        query["llm_common_name"] = llm_id
    if server_type:
        query["server_type"] = server_type

    pod_log = mongo_client.find_one("pod_benchmarks", query)
    
    if not pod_log:
        raise HTTPException(status_code=404, detail="Pod log not found")

    ## Find benchmark result based on pod log
    benchmark_results = mongo_client.find_many("benchmark_results", {"pod_id": pod_log["pod_id"], "benchmark_type": {"$ne": "throughput"}})

    benchmark_result = mongo_client.find_one("benchmark_results", {"pod_id": pod_log["pod_id"], "benchmark_type": "throughput"})
    
    if not benchmark_result:
        raise HTTPException(status_code=404, detail="Benchmark result not found")

    graph_info = []
    
    for result in benchmark_results:
        graph_info.append({
            "x": result.get("rate"),
            "y": result.get("request_latency"),
        })

    comment_counts = mongo_client.find_many("comments", {"pod_id": pod_log.get("pod_id"),"parent_id": None})
    
    return BenchmarkResult(
        pod_id=pod_log.get("pod_id"),
        model_name=pod_log.get("llm_common_name"),
        benchmark_type=benchmark_result.get("benchmark_type"),
        parameter_size=pod_log.get("llm_parameter_size"),
        inference_engine=pod_log.get("inference_name"),
        hardware=pod_log.get("gpu_id"),
        memory=pod_log.get("container_disk_in_gb"),
        download_speed=None, ## todo: Add download speed
        server_type=pod_log.get("server_type"),
        cost=pod_log.get("pod_cost", {}).get("used_balance") if pod_log.get("pod_cost") else None,
        avg_ttft_ms=benchmark_result.get("time_to_first_token_ms"),
        avg_throughput=benchmark_result.get("output_tokens_per_second"),
        test_duration=pod_log.get("benchmark_duration"),
        container_setup=pod_log.get("time_taken_to_start_server"),
        avg_input_token_per_request=benchmark_result.get("prompt_token_count"),
        avg_output_token_per_request=benchmark_result.get("output_token_count"),
        total_requests=benchmark_result.get("total_requests"),
        model_download_time=pod_log.get("time_taken_to_upload_llm"),
        server_cost_per_hour=pod_log.get("pod_cost", {}).get("cost_per_hr") if pod_log.get("pod_cost") else None,
        graph_info=graph_info,
        x_axis_label="Request/S",
        y_axis_label="Request Latency",
        likes=pod_log.get("likes", 0),
        comment_count=len(list(comment_counts)),
        is_upvoted_by_user=current_user.github_id in pod_log.get("user_likes", []) if current_user else False,
        icon_url=pod_log.get("icon_url")
    )

@app.get("/multiple_benchmark_results", response_model=BenchmarkResultList)
@cache_result(ttl_seconds=600, cache_prefix="benchmark")  # Cache for 10 minutes
async def benchmark_results_by_pod_ids(pod_ids: List[str] = Query(), page: int = Query(1, ge=1, description="Page number"), per_page: int = Query(6, ge=1, le=50, description="Items per page")):
    """
    Get benchmark results for multiple pod IDs
    Takes a list of pod_ids and returns benchmark results in the same format as /benchmark_results
    Now supports pagination for large lists of pod_ids
    """
    
    if not pod_ids:
        raise HTTPException(status_code=400, detail="At least one pod_id is required")
    
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Apply pagination to the pod_ids list
    total_requested = len(pod_ids)
    skip_value = (page - 1) * per_page
    end_value = skip_value + per_page
    paginated_pod_ids = pod_ids[skip_value:end_value]
    
    benchmark_data = []
    
    for pod_id in paginated_pod_ids:
        # Find related pod log
        pod_log = mongo_client.find_one("pod_benchmarks", {"pod_id": pod_id})
        
        if not pod_log:
            # Skip non-existent pod logs but don't fail the entire request
            continue
            
        ## Find benchmark result based on pod log
        benchmark_results = mongo_client.find_many("benchmark_results", {"pod_id": pod_log["pod_id"], "benchmark_type": {"$ne": "throughput"}})
        
        benchmark_result = mongo_client.find_one("benchmark_results", {"pod_id": pod_log["pod_id"], "benchmark_type": "throughput"})
        
        if not benchmark_result:
            # Skip pods without benchmark results but don't fail the entire request
            continue
            
        graph_info = []
        
        for result in benchmark_results:
            graph_info.append({
                "x": result.get("rate"),
                "y": result.get("request_latency"),
            })

        comment_counts = mongo_client.find_many("comments", {"pod_id": pod_log.get("pod_id"),"parent_id": None})
        
        benchmark_data.append(BenchmarkResult(
            pod_id=pod_log.get("pod_id"),
            model_name=pod_log.get("llm_common_name"),
            benchmark_type=benchmark_result.get("benchmark_type"),
            parameter_size=pod_log.get("llm_parameter_size"),
            inference_engine=pod_log.get("inference_name"),
            hardware=pod_log.get("gpu_id"),
            memory=pod_log.get("container_disk_in_gb"),
            download_speed=None, ## todo: Add download speed
            server_type=pod_log.get("server_type"),
            cost=pod_log.get("pod_cost", {}).get("used_balance") if pod_log.get("pod_cost") else None,
            avg_ttft_ms=benchmark_result.get("time_to_first_token_ms"),
            avg_throughput=benchmark_result.get("output_tokens_per_second"),
            test_duration=pod_log.get("benchmark_duration"),
            container_setup=pod_log.get("time_taken_to_start_server"),
            avg_input_token_per_request=benchmark_result.get("prompt_token_count"),
            avg_output_token_per_request=benchmark_result.get("output_token_count"),
            total_requests=benchmark_result.get("total_requests"),
            model_download_time=pod_log.get("time_taken_to_upload_llm"),
            server_cost_per_hour=pod_log.get("pod_cost", {}).get("cost_per_hr") if pod_log.get("pod_cost") else None,
            graph_info=graph_info,
            x_axis_label="Rate",
            y_axis_label="Request Latency",
            likes=pod_log.get("likes", 0),
            comment_count=len(list(comment_counts)),
            icon_url=pod_log.get("icon_url")
        ))
    
    # Import pagination utility
    from .utils import create_pagination_meta
    
    # Create pagination metadata based on the total number of requested pod_ids
    pagination_meta = create_pagination_meta(page, per_page, total_requested)
    
    return BenchmarkResultList(
        benchmark_results=benchmark_data,
        pagination=pagination_meta
    )


@app.get("/get_all_benchmarks", response_model=BenchmarkResultList)
@cache_result(ttl_seconds=900, cache_prefix="benchmark")  # Cache for 15 minutes
async def get_all_benchmarks(page: int = Query(1, ge=1, description="Page number"), per_page: int = Query(6, ge=1, le=50, description="Items per page")):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Get total count for pagination
    total_count = mongo_client.count_documents("pod_benchmarks", {})
    
    # Calculate skip value for pagination
    skip_value = (page - 1) * per_page
    
    benchmark_data = []
    pod_logs = mongo_client.find_many("pod_benchmarks", {}).skip(skip_value).limit(per_page)
    
    for pod_log in pod_logs:
        benchmark_result = mongo_client.find_one("benchmark_results", {"pod_id": pod_log.get("pod_id"), "benchmark_type": "throughput"})
        
        if not benchmark_result:
            continue  # Skip pods without benchmark results

        comment_counts = mongo_client.find_many("comments", {"pod_id": pod_log.get("pod_id"),"parent_id": None})
        
        benchmark_data.append(BenchmarkResult(
            pod_id=pod_log.get("pod_id"),
            model_name=pod_log.get("llm_common_name"),
            benchmark_type="throughput",
            parameter_size=pod_log.get("llm_parameter_size"),
            inference_engine=pod_log.get("inference_name"),
            hardware=pod_log.get("gpu_id"),
            memory=pod_log.get("container_disk_in_gb"),
            download_speed=None, ## todo: Add download speed
            server_type=pod_log.get("server_type"),
            cost=pod_log.get("pod_cost", {}).get("used_balance") if pod_log.get("pod_cost") else None,
            avg_ttft_ms=benchmark_result.get("time_to_first_token_ms"),
            avg_throughput=benchmark_result.get("output_tokens_per_second"),
            test_duration=pod_log.get("benchmark_duration"),
            container_setup=pod_log.get("time_taken_to_start_ollama_server"),
            avg_input_token_per_request=benchmark_result.get("prompt_token_count"),
            avg_output_token_per_request=benchmark_result.get("output_token_count"),
            total_requests=benchmark_result.get("total_requests"),
            model_download_time=pod_log.get("time_taken_to_upload_llm"),
            server_cost_per_hour=pod_log.get("pod_cost", {}).get("cost_per_hr") if pod_log.get("pod_cost") else None,
            likes=pod_log.get("likes", 0),
            comment_count=len(list(comment_counts)),
            icon_url=pod_log.get("icon_url")

        ))

    # Import pagination utility
    from .utils import create_pagination_meta
    
    # Create pagination metadata
    pagination_meta = create_pagination_meta(page, per_page, total_count)

    return BenchmarkResultList(
        benchmark_results=benchmark_data,
        pagination=pagination_meta
    )

@app.get("/get_benchmark_results_by_filters", response_model=BenchmarkResultFilterResponse)
@cache_result(ttl_seconds=600, cache_prefix="benchmark")  # Cache for 10 minutes
async def get_benchmark_results_by_filters(model_name: Optional[str] = None, inference_engine: Optional[str] = None, parameter_size: Optional[str] = None, hardware: Optional[List[str]] = Query(None), hardware_price: Optional[float] = None, TTFT_max: Optional[float] = None, TTFT_min: Optional[float] = None, TPS_max: Optional[float] = None, TPS_min: Optional[float] = None, benchmark_type: Optional[str] = None, hourly_cost_min: Optional[float] = None, hourly_cost_max: Optional[float] = None, page: int = Query(1, ge=1, description="Page number"), per_page: int = Query(6, ge=1, le=50, description="Items per page")):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    # Convert zero values to None for specific parameters
    if hourly_cost_min == 0:
        hourly_cost_min = None
    if hourly_cost_max == 0:
        hourly_cost_max = None
    if TTFT_max == 0:
        TTFT_max = None
    if TPS_min == 0:
        TPS_min = None

    # Handle TTFT and TPS filtering separately from benchmark_results collection
    valid_pod_ids = None
    
    
    # Check if we have TTFT or TPS filters
    if TTFT_max is not None or TTFT_min is not None or TPS_max is not None or TPS_min is not None:
        benchmark_results_query = {"benchmark_type": "throughput"}
        
        # Build TTFT filter
        if TTFT_max is not None and TTFT_min is not None:
            benchmark_results_query["time_to_first_token_ms"] = {"$gte": TTFT_min, "$lte": TTFT_max}
        elif TTFT_max is not None:
            benchmark_results_query["time_to_first_token_ms"] = {"$lte": TTFT_max}
        elif TTFT_min is not None:
            benchmark_results_query["time_to_first_token_ms"] = {"$gte": TTFT_min}
        
        # Build TPS filter
        if TPS_max is not None and TPS_min is not None:
            benchmark_results_query["output_tokens_per_second"] = {"$gte": TPS_min, "$lte": TPS_max}
        elif TPS_max is not None:
            benchmark_results_query["output_tokens_per_second"] = {"$lte": TPS_max}
        elif TPS_min is not None:
            benchmark_results_query["output_tokens_per_second"] = {"$gte": TPS_min}
        
        # Get pod_ids that match TTFT/TPS criteria
        benchmark_results = mongo_client.find_many("benchmark_results", benchmark_results_query)
        valid_pod_ids = [result.get("pod_id") for result in benchmark_results if result.get("pod_id")]
        
        # If no benchmark results match the criteria, return empty result
        if not valid_pod_ids:
            from .utils import create_pagination_meta
            pagination_meta = create_pagination_meta(page, per_page, 0)
            return BenchmarkResultFilterResponse(
                benchmark_results=[],
                pagination=pagination_meta
            )
    
    # Build query for pod_benchmarks collection (excluding TTFT/TPS which are handled above)
    query = build_benchmark_query(
        model_name=model_name,
        inference_engine=inference_engine,
        parameter_size=parameter_size,
        hardware=hardware,
        hardware_price=hardware_price,
        TTFT_max=None,  # Don't include these in pod_benchmarks query
        TTFT_min=None,
        TPS_max=None,
        TPS_min=None,
        benchmark_type=benchmark_type,
        hourly_cost_min=hourly_cost_min,
        hourly_cost_max=hourly_cost_max
    )
    
    # Add pod_id filter if we have TTFT/TPS constraints
    if valid_pod_ids is not None:
        if len(valid_pod_ids) == 0:
            # No valid pod_ids means no results
            from .utils import create_pagination_meta
            pagination_meta = create_pagination_meta(page, per_page, 0)
            return BenchmarkResultFilterResponse(
                benchmark_results=[],
                pagination=pagination_meta
            )
        else:
            query["pod_id"] = {"$in": valid_pod_ids}

    # Get total count for pagination
    total_count = mongo_client.count_documents("pod_benchmarks", query)
    
    # Calculate skip value for pagination
    skip_value = (page - 1) * per_page
    
    benchmark_data = []
    pod_logs = mongo_client.find_many("pod_benchmarks", query).skip(skip_value).limit(per_page)
    
    for pod_log in pod_logs:
        comment_counts = mongo_client.find_many("comments", {"pod_id": pod_log.get("pod_id"),"parent_id": None})
        benchmark_data.append({
            "pod_id":pod_log.get("pod_id"),
            "model_name":pod_log.get("llm_common_name"),
            "parameter_size":pod_log.get("llm_parameter_size"),
            "inference_engine":pod_log.get("inference_name"),
            "hardware":pod_log.get("gpu_id"),
            "likes":pod_log.get("likes"),
            "comment_count":len(list(comment_counts)),
            "icon_url":pod_log.get("icon_url")
        })

    # Import pagination utility
    from .utils import create_pagination_meta
    
    # Create pagination metadata
    pagination_meta = create_pagination_meta(page, per_page, total_count)

    return BenchmarkResultFilterResponse(
        benchmark_results=benchmark_data,
        pagination=pagination_meta
    )

@app.post("/get_response_from_ai", response_model=ChatResponse)
@limiter.limit("10/minute")
async def get_response_from_ai(request: Request, chat_input: ChatMessageInput, current_user: User = Depends(get_current_user)):
    """
    Get AI response and log chat data to MongoDB - requires GitHub authentication
    """
    ai_client = BenchmarkAIClient()
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    benchmark_data = ""
    
    if chat_input.pod_id:
        pod_log = mongo_client.find_one("pod_benchmarks", {"pod_id": chat_input.pod_id})
        
        if not pod_log:
            raise HTTPException(status_code=404, detail="Pod log not found")

        ## Find benchmark result based on pod log
        benchmark_results = mongo_client.find_many("benchmark_results", {"pod_id": pod_log["pod_id"], "benchmark_type": {"$ne": "throughput"}})

        benchmark_result = mongo_client.find_one("benchmark_results", {"pod_id": pod_log["pod_id"], "benchmark_type": "throughput"})
        
        if not benchmark_result:
            raise HTTPException(status_code=404, detail="Benchmark result not found")

        graph_info = []
        
        for result in benchmark_results:
            graph_info.append({
                "x": result.get("rate"),
                "y": result.get("request_latency"),
            })

        benchmark_data ={
        "pod_id":pod_log.get("pod_id"),
        "model_name":pod_log.get("llm_common_name"),
        "benchmark_type":benchmark_result.get("benchmark_type"),
        "parameter_size":pod_log.get("llm_parameter_size"),
        "inference_engine":pod_log.get("inference_name"),
        "hardware":pod_log.get("gpu_id"),
        "memory":pod_log.get("container_disk_in_gb"),
        "download_speed":None, ## todo: Add download speed
        "server_type":pod_log.get("server_type"),
        "cost":pod_log.get("pod_cost", {}).get("used_balance") if pod_log.get("pod_cost") else None,
        "avg_ttft_ms":benchmark_result.get("time_to_first_token_ms"),
        "avg_throughput":benchmark_result.get("output_tokens_per_second"),
        "test_duration":pod_log.get("benchmark_duration"),
        "container_setup":pod_log.get("time_taken_to_start_server"),
        "avg_input_token_per_request":benchmark_result.get("prompt_token_count"),
        "avg_output_token_per_request":benchmark_result.get("output_token_count"),
        "total_requests":benchmark_result.get("total_requests"),
        "model_download_time":pod_log.get("time_taken_to_upload_llm"),
        "server_cost_per_hour":pod_log.get("pod_cost", {}).get("cost_per_hr") if pod_log.get("pod_cost") else None,
        "graph_info":graph_info,
        "x_axis_label":"Rate",
        "y_axis_label":"Request Latency",
        }

        benchmark_data = str(benchmark_data)

    gpu_pricing_data = ""
    if chat_input.pod_id:
        gpu_pricing_data = get_gpu_prices_by_pod_id_util(chat_input.pod_id)
        gpu_pricing_data = str(gpu_pricing_data)


    # Get AI response
    ai_response = ai_client.chat(message=chat_input.message, benchmark_data=benchmark_data,gpu_pricing_data=gpu_pricing_data)
    timestamp = datetime.utcnow()
    
    # Prepare chat log data
    chat_log_data = {
        "user_message": chat_input.message,
        "ai_response": ai_response,
        "timestamp": timestamp,
        "pod_id": chat_input.pod_id,
        "is_authenticated": True,
        "user_id": current_user.github_id,
        "username": current_user.username,
        "user_email": current_user.email,
        "user_avatar": current_user.avatar_url
    }
    
    # Save to MongoDB
    mongo_client.insert_one("chat_logs", chat_log_data)
    
    return ChatResponse(
        response=ai_response,
        timestamp=timestamp,
        log_id=str(chat_log_data.get("_id")),
        pod_id=chat_input.pod_id,
        user_authenticated=True,
        user_info=str(current_user.github_id)
    )

@app.post("/get_response_from_ai/stream")
@limiter.limit("5/minute")
async def get_response_from_ai_stream(request: Request, chat_input: ChatMessageInput, current_user: User = Depends(get_current_user)):
    """
    Get AI response with streaming - requires GitHub authentication
    Returns Server-Sent Events (SSE) stream with real-time AI response
    """
    ai_client = BenchmarkAIClient()
    mongo_client = Mongo(os.getenv("MONGODB_URL"))

    benchmark_data = ""
    
    if chat_input.pod_id:
        pod_log = mongo_client.find_one("pod_benchmarks", {"pod_id": chat_input.pod_id})
        
        if not pod_log:
            raise HTTPException(status_code=404, detail="Pod log not found")

        ## Find benchmark result based on pod log
        benchmark_results = mongo_client.find_many("benchmark_results", {"pod_id": pod_log["pod_id"], "benchmark_type": {"$ne": "throughput"}})

        benchmark_result = mongo_client.find_one("benchmark_results", {"pod_id": pod_log["pod_id"], "benchmark_type": "throughput"})
        
        if not benchmark_result:
            raise HTTPException(status_code=404, detail="Benchmark result not found")

        graph_info = []
        
        for result in benchmark_results:
            graph_info.append({
                "x": result.get("rate"),
                "y": result.get("request_latency"),
            })

        benchmark_data = {
            "pod_id": pod_log.get("pod_id"),
            "model_name": pod_log.get("llm_common_name"),
            "benchmark_type": benchmark_result.get("benchmark_type"),
            "parameter_size": pod_log.get("llm_parameter_size"),
            "inference_engine": pod_log.get("inference_name"),
            "hardware": pod_log.get("gpu_id"),
            "memory": pod_log.get("container_disk_in_gb"),
            "download_speed": None,  ## todo: Add download speed
            "server_type": pod_log.get("server_type"),
            "cost": pod_log.get("pod_cost", {}).get("used_balance") if pod_log.get("pod_cost") else None,
            "avg_ttft_ms": benchmark_result.get("time_to_first_token_ms"),
            "avg_throughput": benchmark_result.get("output_tokens_per_second"),
            "test_duration": pod_log.get("benchmark_duration"),
            "container_setup": pod_log.get("time_taken_to_start_server"),
            "avg_input_token_per_request": benchmark_result.get("prompt_token_count"),
            "avg_output_token_per_request": benchmark_result.get("output_token_count"),
            "total_requests": benchmark_result.get("total_requests"),
            "model_download_time": pod_log.get("time_taken_to_upload_llm"),
            "server_cost_per_hour": pod_log.get("pod_cost", {}).get("cost_per_hr") if pod_log.get("pod_cost") else None,
            "graph_info": graph_info,
            "x_axis_label": "Rate",
            "y_axis_label": "Request Latency",
        }

        benchmark_data = str(benchmark_data)

    gpu_pricing_data = ""
    if chat_input.pod_id:
        gpu_pricing_data = get_gpu_prices_by_pod_id_util(chat_input.pod_id)
        gpu_pricing_data = str(gpu_pricing_data)

    # Prepare chat log data structure
    timestamp = datetime.utcnow()
    chat_log_data = {
        "user_message": chat_input.message,
        "ai_response": "",  # Will be built incrementally
        "timestamp": timestamp,
        "pod_id": chat_input.pod_id,
        "is_authenticated": True,
        "user_id": current_user.github_id,
        "username": current_user.username,
        "user_email": current_user.email,
        "user_avatar": current_user.avatar_url
    }

    async def generate_sse_stream():
        """Generate Server-Sent Events stream"""
        complete_response = ""
        
        try:
            # Send initial connection event
            yield f"data: {json_module.dumps({'type': 'start', 'message': 'Connection established'})}\n\n"
            
            # Stream AI response
            async for chunk in ai_client.chat_stream(
                message=chat_input.message, 
                benchmark_data=benchmark_data,
                gpu_pricing_data=gpu_pricing_data
            ):
                complete_response += chunk
                
                # Send content chunk
                yield f"data: {json_module.dumps({'type': 'content', 'chunk': chunk})}\n\n"
                
                # Add small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            # Send completion event with metadata
            completion_data = {
                "type": "done",
                "timestamp": timestamp.isoformat(),
                "pod_id": chat_input.pod_id,
                "user_authenticated": True,
                "user_info": str(current_user.github_id),
                "complete_response": complete_response
            }
            yield f"data: {json_module.dumps(completion_data)}\n\n"
            
            # Save complete conversation to MongoDB
            chat_log_data["ai_response"] = complete_response
            mongo_client.insert_one("chat_logs", chat_log_data)
            
            # Send final log event with database ID
            log_data = {
                "type": "logged",
                "log_id": str(chat_log_data.get("_id"))
            }
            yield f"data: {json_module.dumps(log_data)}\n\n"
            
        except Exception as e:
            # Send error event
            error_data = {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json_module.dumps(error_data)}\n\n"
            
            # Still try to log partial response if we have any
            if complete_response:
                chat_log_data["ai_response"] = complete_response + f"\n[ERROR: {str(e)}]"
                try:
                    mongo_client.insert_one("chat_logs", chat_log_data)
                except:
                    pass  # Ignore logging errors during error handling

    return StreamingResponse(
        generate_sse_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*"
        }
    )

@app.post("/upvote_pod", response_model=APIResponse)
async def upvote_pod(upvote_input: PodUpvoteInput, current_user: User = Depends(get_current_user)):
    """
    Upvote a pod - requires GitHub authentication
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    pod_log = mongo_client.find_one("pod_benchmarks", {"pod_id": upvote_input.pod_id})
    
    if not pod_log:
        raise HTTPException(status_code=404, detail="Pod not found")

    # Check if user already liked this pod
    user_likes = pod_log.get("user_likes", [])
    if current_user.github_id in user_likes:
        raise HTTPException(status_code=400, detail="User already liked this pod")
    
    # Add like and track user who liked
    mongo_client.update_one(
        "pod_benchmarks",
        {"pod_id": upvote_input.pod_id},
        {"$inc": {"likes": 1}, "$push": {"user_likes": current_user.github_id}}
    )
    
    # Invalidate related cache entries
    invalidate_benchmark_cache(upvote_input.pod_id)
    
    return APIResponse(
        success=True,
        message="Pod upvoted successfully",
        data=None
    )

@app.post("/remove_upvote_from_pod", response_model=APIResponse)
async def remove_upvote_from_pod(upvote_input: PodUpvoteInput, current_user: User = Depends(get_current_user)):
    """
    Remove upvote from a pod - requires GitHub authentication
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    pod_log = mongo_client.find_one("pod_benchmarks", {"pod_id": upvote_input.pod_id})
    
    if not pod_log:
        raise HTTPException(status_code=404, detail="Pod not found")

    # Check if user already liked this pod
    user_likes = pod_log.get("user_likes", [])
    if current_user.github_id not in user_likes:
        raise HTTPException(status_code=400, detail="User can not upvote this pod")

    # Remove like and track user who unliked
    mongo_client.update_one(
        "pod_benchmarks",
        {"pod_id": upvote_input.pod_id},
        {"$inc": {"likes": -1}, "$pull": {"user_likes": current_user.github_id}}
    )
    
    # Invalidate related cache entries
    invalidate_benchmark_cache(upvote_input.pod_id)
    
    return APIResponse(
        success=True,
        message="Upvote removed successfully",
        data=None
    )

@app.post("/post_comment", response_model=APIResponse)
@limiter.limit("30/minute")
async def post_comment(request: Request, comment_input: CommentInput, current_user: User = Depends(get_current_user)):
    """
    Post a comment - requires GitHub authentication
    Supports parent-child relationship (max 1 level deep)
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # AI-based content moderation
    ai_client = BenchmarkAIClient()
    
    moderation_prompt = f"""
    You are a content moderation AI. Analyze the following comment and determine if it contains:
    1. Harmful language (threats, harassment, hate speech)
    2. Curse words or profanity
    3. Inappropriate content for a professional technical community
    
    Comment to analyze: "{comment_input.comment}"
    
    Respond with only one word:
    - "APPROVED" if the comment is appropriate and professional
    - "REJECTED" if the comment contains harmful language, curse words, or inappropriate content
    
    Be strict but fair - this is for a technical AI/ML community discussion.
    """
    
    try:
        moderation_result = ai_client.chat(message=moderation_prompt)
        
        # Clean up the response and check result
        moderation_decision = moderation_result.strip().upper()
        
        if "REJECTED" in moderation_decision:
            raise HTTPException(
                status_code=406,  # Unprocessable Entity
                detail={
                    "error_type": "CONTENT_MODERATION_FAILED",
                    "message": "Your comment contains inappropriate content. Please revise your comment to be more professional and respectful.",
                    "field": "comment"
                }
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions (our moderation failure)
        raise
    except Exception as e:
        # If AI moderation fails, log the error but don't block the comment
        # You might want to add logging here
        print(f"Content moderation failed: {str(e)}")
        # Continue with posting the comment if AI service is down
    
    # Check if parent comment exists and is valid
    parent_id = None
    depth = 0
    
    if comment_input.parent_id:
        parent_comment = mongo_client.find_one("comments", {"_id": ObjectId(comment_input.parent_id)})
        if not parent_comment:
            raise HTTPException(status_code=404, detail="Parent comment not found")
        
        # Check if parent is already a reply (max 1 level deep)
        if parent_comment.get("parent_id"):
            raise HTTPException(status_code=400, detail="Cannot reply to a reply. Only one level of nesting is allowed.")
        
        parent_id = comment_input.parent_id
        depth = 1
    
    comment_data = {
        "pod_id": comment_input.pod_id,
        "comment": comment_input.comment,
        "likes": 0,
        "created_at": datetime.utcnow(),
        "user_likes": [],  # Initialize empty array for tracking likes
        "user_id": current_user.github_id,
        "username": current_user.username,
        "user_email": current_user.email,
        "user_avatar": current_user.avatar_url,
        "parent_id": parent_id,
        "depth": depth,
        "ranking_score": 0.0  # Will be calculated based on likes and time
    }
    
    mongo_client.insert_one("comments", comment_data)
    comment_id = str(comment_data.get("_id"))
    
    # Update parent comment to indicate it has replies
    if parent_id:
        mongo_client.update_one(
            "comments",
            {"_id": ObjectId(parent_id)},
            {"$inc": {"reply_count": 1}}
        )
    
    # Invalidate comment cache entries
    invalidate_comments_cache(comment_input.pod_id)
    
    return APIResponse(
        success=True,
        message="Comment posted successfully",
        data={
            "comment_id": comment_id,
            "pod_id": comment_input.pod_id,
            "comment": comment_input.comment,
            "user_id": current_user.github_id,
            "username": current_user.username,
            "parent_id": parent_id,
            "depth": depth
        }
    )

@app.get("/get_comments", response_model=GetCommentsResponse)
@cache_result(ttl_seconds=300, cache_prefix="comments")  # Cache for 5 minutes
async def get_comments(pod_id: str, current_user: Optional[User] = Depends(get_current_user_optional)):
    """
    Get comments for a specific pod - authentication optional for viewing
    Returns hierarchical structure with ranking
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    comments = mongo_client.find_many("comments", {"pod_id": pod_id})
    print(current_user)
    
    # Separate parent and child comments
    parent_comments = []
    child_comments = []
    
    for comment in comments:
        # Calculate ranking score
        ranking_score = calculate_ranking_score(
            comment.get("likes", 0),
            comment.get("created_at"),
            comment.get("reply_count", 0)
        )

        
        comment_data = {
            "_id": str(comment.get("_id")),
            "comment": comment.get("comment"),
            "pod_id": comment.get("pod_id"),
            "likes": comment.get("likes", 0),
            "created_at": comment.get("created_at"),
            "user_id": comment.get("user_id"),
            "username": comment.get("username"),
            "user_email": comment.get("user_email"),
            "user_avatar": comment.get("user_avatar"),
            "parent_id": comment.get("parent_id"),
            "depth": comment.get("depth", 0),
            "reply_count": comment.get("reply_count", 0),
            "ranking_score": ranking_score,
            "user_has_liked": False
        }
        
        
        # Add user's like status if authenticated
        if current_user and comment.get("user_likes"):
            user_likes = comment.get("user_likes", [])
            comment_data["user_has_liked"] = current_user.github_id in user_likes
        
        # Determine if it's a parent or child comment
        if comment.get("parent_id"):
            child_comments.append(comment_data)
        else:
            comment_data["is_parent"] = True
            comment_data["has_replies"] = comment.get("reply_count", 0) > 0
            parent_comments.append(comment_data)
    
    # Sort parent comments by ranking score (descending)
    parent_comments.sort(key=lambda x: x["ranking_score"], reverse=True)
    
    # Sort child comments by creation time (oldest first)
    child_comments.sort(key=lambda x: x["created_at"])
    
    # Create hierarchical structure
    for parent in parent_comments:
        parent["replies"] = [
            child for child in child_comments 
            if child["parent_id"] == parent["_id"]
        ]

    

    return GetCommentsResponse(
        parent_comments=parent_comments,
        total_comments=len(parent_comments) + len(child_comments),
        total_parents=len(parent_comments),
        total_replies=len(child_comments)
    )

@app.get("/get_available_benchmark_options", response_model=GetAvailableBenchmarkOptionsResponse)
@cache_result(ttl_seconds=1800, cache_prefix="benchmark")  # Cache for 30 minutes - these options change rarely
async def get_available_benchmark_options():
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    available_benchmark_options = mongo_client.find_many("pod_benchmarks", {})
    available_benchmark_options = [pod_benchmark for pod_benchmark in available_benchmark_options]
    ## Find unique model_name, inference_engine, parameter_size, hardware, server_type
    unique_model_names = list(set(pod_benchmark.get("llm_common_name") for pod_benchmark in available_benchmark_options))
    unique_inference_engines = list(set(pod_benchmark.get("inference_name") for pod_benchmark in available_benchmark_options))
    unique_parameter_sizes = list(set(pod_benchmark.get("llm_parameter_size") for pod_benchmark in available_benchmark_options))
    unique_hardwares = list(set(pod_benchmark.get("gpu_id") for pod_benchmark in available_benchmark_options))
    unique_server_types = list(set(pod_benchmark.get("server_type") for pod_benchmark in available_benchmark_options))

    return GetAvailableBenchmarkOptionsResponse(
        model_names=unique_model_names,
        inference_engines=unique_inference_engines,
        parameter_sizes=unique_parameter_sizes,
        hardwares=unique_hardwares,
        server_types=unique_server_types
    )
    

@app.get("/get_all_comments", response_model=GetAllCommentsResponse)
async def get_all_comments(current_user: Optional[User] = Depends(get_current_user_optional)):
    """
    Get all comments - authentication optional for viewing
    Returns flat list with ranking information
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    comments = mongo_client.find_many("comments", {})
    comment_list = []
    
    for comment in comments:
        # Calculate ranking score
        ranking_score = calculate_ranking_score(
            comment.get("likes", 0),
            comment.get("created_at"),
            comment.get("reply_count", 0)
        )
        
        comment_data = {
            "_id": str(comment.get("_id")),
            "comment": comment.get("comment"),
            "likes": comment.get("likes", 0),
            "created_at": comment.get("created_at"),
            "pod_id": comment.get("pod_id"),
            "user_id": comment.get("user_id"),
            "username": comment.get("username"),
            "user_email": comment.get("user_email"),
            "user_avatar": comment.get("user_avatar"),
            "parent_id": comment.get("parent_id"),
            "depth": comment.get("depth", 0),
            "reply_count": comment.get("reply_count", 0),
            "ranking_score": ranking_score,
            "is_parent": not comment.get("parent_id"),
            "has_replies": comment.get("reply_count", 0) > 0,
            "user_has_liked": False
        }
        
        # Add user's like status if authenticated
        if current_user and comment.get("user_likes"):
            user_likes = comment.get("user_likes", [])
            comment_data["user_has_liked"] = current_user.github_id in user_likes
        
        comment_list.append(comment_data)
    
    # Sort by ranking score (descending)
    comment_list.sort(key=lambda x: x["ranking_score"], reverse=True)
    
    return GetAllCommentsResponse(
        comments=comment_list
    )
@app.post("/add_like", response_model=LikeResponse)
async def add_like(like_input: LikeInput, current_user: User = Depends(get_current_user)):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Check if comment exists
    comment = mongo_client.find_one("comments", {"_id": ObjectId(like_input.comment_id)})
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    # Check if user already liked this comment
    user_likes = comment.get("user_likes", [])
    if current_user.github_id in user_likes:
        raise HTTPException(status_code=400, detail="User already liked this comment")
    
    # Add like and track user who liked
    mongo_client.update_one(
        "comments",
        {"_id": ObjectId(like_input.comment_id)},
        {
            "$inc": {"likes": 1},
            "$push": {"user_likes": current_user.github_id}
        }
    )
    
    # Update ranking score
    new_likes = comment.get("likes", 0) + 1
    new_ranking_score = calculate_ranking_score(
        new_likes,
        comment.get("created_at"),
        comment.get("reply_count", 0)
    )
    
    mongo_client.update_one(
        "comments",
        {"_id": ObjectId(like_input.comment_id)},
        {"$set": {"ranking_score": new_ranking_score}}
    )
    
    # Invalidate comment cache entries
    invalidate_comments_cache(comment.get("pod_id"))
    
    return LikeResponse(
        comment_id=like_input.comment_id,
        user_id=current_user.github_id,
        username=current_user.username,
        total_likes=new_likes,
        ranking_score=new_ranking_score
    )

@app.post("/remove_like", response_model=RemoveLikeResponse)
async def remove_like(like_input: LikeInput, current_user: User = Depends(get_current_user)):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Check if comment exists
    comment = mongo_client.find_one("comments", {"_id": ObjectId(like_input.comment_id)})
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    # Check if user liked this comment
    user_likes = comment.get("user_likes", [])
    if current_user.github_id not in user_likes:
        raise HTTPException(status_code=400, detail="User has not liked this comment")
    
    # Remove like and track user who unliked
    mongo_client.update_one(
        "comments",
        {"_id": ObjectId(like_input.comment_id)},
        {
            "$inc": {"likes": -1},
            "$pull": {"user_likes": current_user.github_id}
        }
    )
    
    # Update ranking score
    new_likes = comment.get("likes", 0) - 1
    new_ranking_score = calculate_ranking_score(
        new_likes,
        comment.get("created_at"),
        comment.get("reply_count", 0)
    )
    
    mongo_client.update_one(
        "comments",
        {"_id": ObjectId(like_input.comment_id)},
        {"$set": {"ranking_score": new_ranking_score}}
    )
    
    # Invalidate comment cache entries
    invalidate_comments_cache(comment.get("pod_id"))
    
    return RemoveLikeResponse(
        comment_id=like_input.comment_id,
        user_id=current_user.github_id,
        username=current_user.username,
        total_likes=new_likes,
        ranking_score=new_ranking_score
    )

@app.delete("/delete_comment", response_model=DeleteCommentResponse)
async def delete_comment(like_input: LikeInput, current_user: User = Depends(get_current_user)):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Check if comment exists
    comment = mongo_client.find_one("comments", {"_id": ObjectId(like_input.comment_id)})
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    # Check if user owns this comment or is admin (you can add admin logic here)
    if comment.get("user_id") != current_user.github_id:
        raise HTTPException(status_code=403, detail="You can only delete your own comments")
    
    # If this is a reply, update parent's reply count
    if comment.get("parent_id"):
        mongo_client.update_one(
            "comments",
            {"_id": ObjectId(comment.get("parent_id"))},
            {"$inc": {"reply_count": -1}}
        )
    
    # Delete the comment
    mongo_client.delete_one("comments", {"_id": ObjectId(like_input.comment_id)})
    
    # Invalidate comment cache entries
    invalidate_comments_cache(comment.get("pod_id"))
    
    return DeleteCommentResponse(
        comment_id=like_input.comment_id
    )

@app.get("/get_user_comments", response_model=GetUserCommentsResponse)
async def get_user_comments(current_user: User = Depends(get_current_user)):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    comments = mongo_client.find_many("comments", {"user_id": current_user.github_id})
    comment_list = []
    
    for comment in comments:
        comment_data = {
            "_id": str(comment.get("_id")),
            "comment": comment.get("comment"),
            "likes": comment.get("likes"),
            "created_at": comment.get("created_at"),
            "pod_id": comment.get("pod_id"),
            "username": comment.get("username"),
            "user_email": comment.get("user_email"),
            "user_avatar": comment.get("user_avatar")
        }
        comment_list.append(comment_data)
    
    return GetUserCommentsResponse(
        comments=comment_list
    )

@app.get("/get_chat_logs", response_model=GetChatLogsResponse)
async def get_chat_logs(current_user: User = Depends(get_current_user)):
    """
    Get all chat logs - requires GitHub authentication
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    chat_logs = mongo_client.find_many("chat_logs", {})
    log_list = []
    
    for log in chat_logs:
        log_data = {
            "_id": str(log.get("_id")),
            "user_message": log.get("user_message"),
            "ai_response": log.get("ai_response"),
            "timestamp": log.get("timestamp"),
            "pod_id": log.get("pod_id"),
            "is_authenticated": log.get("is_authenticated", True),
            "user_id": log.get("user_id"),
            "username": log.get("username"),
            "user_email": log.get("user_email"),
            "user_avatar": log.get("user_avatar")
        }
        log_list.append(log_data)
    
    return GetChatLogsResponse(
        logs=log_list
    )

@app.get("/get_user_chat_logs", response_model=GetChatLogsResponse)
async def get_user_chat_logs(current_user: User = Depends(get_current_user)):
    """
    Get chat logs for the authenticated user
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    chat_logs = mongo_client.find_many("chat_logs", {"user_id": current_user.github_id})
    log_list = []
    
    for log in chat_logs:
        log_data = {
            "_id": str(log.get("_id")),
            "user_message": log.get("user_message"),
            "ai_response": log.get("ai_response"),
            "timestamp": log.get("timestamp"),
            "pod_id": log.get("pod_id"),
            "is_authenticated": log.get("is_authenticated", True),
            "user_id": log.get("user_id"),
            "username": log.get("username"),
            "user_email": log.get("user_email"),
            "user_avatar": log.get("user_avatar")
        }
        log_list.append(log_data)
    
    return GetChatLogsResponse(
        logs=log_list
    )

@app.get("/get_pod_chat_logs", response_model=GetPodChatLogsResponse)
async def get_pod_chat_logs(pod_id: str, current_user: User = Depends(get_current_user)):
    """
    Get chat logs for a specific pod - requires GitHub authentication
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    chat_logs = mongo_client.find_many("chat_logs", {"pod_id": pod_id,"user_id":current_user.github_id})
    log_list = []
    
    for log in chat_logs:
        log_data = {
            "_id": str(log.get("_id")),
            "user_message": log.get("user_message"),
            "ai_response": log.get("ai_response"),
            "timestamp": log.get("timestamp"),
            "pod_id": log.get("pod_id"),
            "is_authenticated": log.get("is_authenticated", True),
            "user_id": log.get("user_id"),
            "username": log.get("username"),
            "user_email": log.get("user_email"),
            "user_avatar": log.get("user_avatar")
        }
        log_list.append(log_data)
    
    return GetPodChatLogsResponse(
        logs=log_list
    )

@app.delete("/delete_chat_log", response_model=DeleteChatLogResponse)
async def delete_chat_log(delete_input: ChatLogDeleteInput, current_user: User = Depends(get_current_user)):
    """
    Delete a specific chat log - only owner can delete
    """
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    # Check if chat log exists
    chat_log = mongo_client.find_one("chat_logs", {"_id": ObjectId(delete_input.log_id)})
    if not chat_log:
        raise HTTPException(status_code=404, detail="Chat log not found")
    
    # Check if user owns this chat log
    if chat_log.get("user_id") != current_user.github_id:
        raise HTTPException(status_code=403, detail="You can only delete your own chat logs")
    
    # Delete the chat log
    mongo_client.delete_one("chat_logs", {"_id": ObjectId(delete_input.log_id)})
    
    return DeleteChatLogResponse(
        log_id=delete_input.log_id
    )

# @app.get("/get_all_likes", response_model=GetAllLikesResponse)
# async def get_all_likes():
#     mongo_client = Mongo(os.getenv("MONGODB_URL"))
#     likes = mongo_client.find_many("comments", {})
#     likes_list = []
#     for like in likes:
#         likes_list.append({
#             "comment_id": like.get("comment_id"),
#             "user_id": like.get("user_id"),
#             "username": like.get("username"),
#         })

#     return GetAllLikesResponse(
#         likes=likes_list
#     )

@app.get("/get_gpu_prices_by_platform", response_model=GetGpuPricesByPlatformResponse)
@cache_result(ttl_seconds=1800, cache_prefix="gpu")  # Cache for 30 minutes
async def get_gpu_prices_by_platform(platform: Optional[str] = None, gpu_name: Optional[str] = None):
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    if platform:
        gpu_prices = mongo_client.find_many("gpu_pricing_processed", {"platform": platform})
    elif gpu_name:
        gpu_prices = mongo_client.find_many("gpu_pricing_processed", {"normalized_name": gpu_name})
    else:
        gpu_prices = mongo_client.find_many("gpu_pricing_processed", {})

    gpu_prices_list = []
    for gpu_price in gpu_prices:
        gpu_prices_list.append({
            "name": gpu_price.get("normalized_name"),
            "price": gpu_price.get("price"),
            "platform": gpu_price.get("platform"),
        })
    return GetGpuPricesByPlatformResponse(
        gpu_prices=gpu_prices_list
    )

@app.get("/get_gpu_prices_by_pod_id", response_model=GetGpuPricesByPlatformResponse)
async def get_gpu_prices_by_pod_id(pod_id: str):
    
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    pod_benchmark = mongo_client.find_one("pod_benchmarks", {"pod_id": pod_id})
    
    if not pod_benchmark:
        raise HTTPException(status_code=404, detail="Pod not found")

    pod_gpu = pod_benchmark.get("gpu_id")
    if not pod_gpu:
        raise HTTPException(status_code=404, detail="GPU information not found for this pod")
        
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
                raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")

    for gpu_price in gpu_prices_list:
        gpu_price["platform"] = gpu_price["platform"].capitalize()
        if gpu_price.get("platform") == "Primeintellect":
            gpu_price["platform"] = "Prime Intellect"
        elif gpu_price.get("platform") == "Akash":
            gpu_price["platform"] = "Akash Network"
    return GetGpuPricesByPlatformResponse(
        gpu_prices=gpu_prices_list
    )

# Cache management endpoints
@app.post("/admin/cache/clear", response_model=APIResponse)
async def clear_cache():
    """Clear all cache entries - admin only"""
    # You might want to add admin role check here
    from .cache import cache_service
    
    success = cache_service.clear_all()
    
    return APIResponse(
        success=success,
        message="Cache cleared successfully" if success else "Failed to clear cache",
        data=None
    )

@app.post("/admin/cache/invalidate/benchmark", response_model=APIResponse)
async def invalidate_benchmark_cache_endpoint(pod_id: Optional[str] = None, current_user: User = Depends(get_current_user)):
    """Invalidate benchmark cache entries"""
    deleted_count = invalidate_benchmark_cache(pod_id)
    
    return APIResponse(
        success=True,
        message=f"Invalidated {deleted_count} benchmark cache entries",
        data={"deleted_count": deleted_count}
    )

@app.post("/admin/cache/invalidate/comments", response_model=APIResponse)
async def invalidate_comments_cache_endpoint(pod_id: Optional[str] = None, current_user: User = Depends(get_current_user)):
    """Invalidate comment cache entries"""
    deleted_count = invalidate_comments_cache(pod_id)
    
    return APIResponse(
        success=True,
        message=f"Invalidated {deleted_count} comment cache entries",
        data={"deleted_count": deleted_count}
    )

@app.post("/admin/cache/invalidate/gpu", response_model=APIResponse)
async def invalidate_gpu_cache_endpoint(current_user: User = Depends(get_current_user)):
    """Invalidate GPU pricing cache entries"""
    deleted_count = invalidate_gpu_cache()
    
    return APIResponse(
        success=True,
        message=f"Invalidated {deleted_count} GPU cache entries",
        data={"deleted_count": deleted_count}
    )

    
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    # Handle both string and dict details
    if isinstance(exc.detail, dict):
        # For structured error responses (like content moderation)
        message = exc.detail.get("message", "An error occurred")
        data = {
            "status_code": exc.status_code,
            "error_detail": exc.detail
        }
    else:
        # For simple string error messages
        message = exc.detail
        data = {"status_code": exc.status_code}
    
    # Return a proper JSON response
    response_content = {
        "success": False,
        "message": message,
        "data": data
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content
    )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )





















# # Token management endpoints
# @app.post("/auth/tokens", response_model=APIResponse)
# async def save_token(token_request: TokenUpdateRequest, current_user: User = Depends(get_current_user)):
#     """Save or update a token for the current user"""
#     auth_service = AuthService()
    
#     # Calculate expiration time if provided
#     expires_at = None
#     if token_request.expires_at:
#         expires_at = token_request.expires_at
    
#     auth_service.save_user_tokens(
#         github_id=current_user.github_id,
#         token_type=token_request.token_type,
#         token_value=token_request.token_value,
#         expires_at=expires_at
#     )
    
#     return APIResponse(
#         success=True,
#         message=f"Successfully saved {token_request.token_type} token",
#         data={"token_type": token_request.token_type}
#     )

# @app.get("/auth/tokens", response_model=APIResponse)
# async def get_user_tokens(current_user: User = Depends(get_current_user)):
#     """Get all tokens for the current user"""
#     auth_service = AuthService()
#     user_tokens = auth_service.get_user_tokens(current_user.github_id)
    
#     if not user_tokens:
#         return APIResponse(
#             success=True,
#             message="No tokens found for user",
#             data={"tokens": {}}
#         )
    
#     # Return token status without exposing actual token values
#     token_status = {}
#     for field in user_tokens.__fields__:
#         if field.endswith('_token') or field.endswith('_key'):
#             token_type = field.replace('_token', '').replace('_key', '')
#             token_value = getattr(user_tokens, field)
#             expires_field = f"{token_type}_token_expires_at"
#             expires_at = getattr(user_tokens, expires_field, None) if hasattr(user_tokens, expires_field) else None
            
#             token_status[token_type] = {
#                 "has_token": bool(token_value),
#                 "expires_at": expires_at.isoformat() if expires_at else None,
#                 "is_valid": auth_service.is_token_valid(current_user.github_id, token_type) if token_value else False
#             }
    
#     return APIResponse(
#         success=True,
#         message="User tokens retrieved successfully",
#         data={"tokens": token_status}
#     )

# @app.get("/auth/tokens/{token_type}", response_model=APIResponse)
# async def get_token_status(token_type: str, current_user: User = Depends(get_current_user)):
#     """Get status of a specific token type"""
#     auth_service = AuthService()
    
#     token_value = auth_service.get_user_token(current_user.github_id, token_type)
#     user_tokens = auth_service.get_user_tokens(current_user.github_id)
    
#     expires_at = None
#     if user_tokens:
#         expires_field = f"{token_type}_token_expires_at"
#         expires_at = getattr(user_tokens, expires_field, None) if hasattr(user_tokens, expires_field) else None
    
#     token_response = TokenResponse(
#         token_type=token_type,
#         has_token=bool(token_value),
#         expires_at=expires_at
#     )
    
#     return APIResponse(
#         success=True,
#         message=f"Token status for {token_type}",
#         data=token_response.dict()
#     )

# @app.delete("/auth/tokens/{token_type}", response_model=APIResponse)
# async def delete_token(token_type: str, current_user: User = Depends(get_current_user)):
#     """Delete a specific token for the current user"""
#     auth_service = AuthService()
#     auth_service.delete_user_token(current_user.github_id, token_type)
    
#     return APIResponse(
#         success=True,
#         message=f"Successfully deleted {token_type} token",
#         data={"token_type": token_type}
#     )

# @app.get("/auth/tokens/{token_type}/value", response_model=APIResponse)
# async def get_token_value(token_type: str, current_user: User = Depends(get_current_user)):
#     """Get the actual token value (use with caution)"""
#     auth_service = AuthService()
#     token_value = auth_service.get_user_token(current_user.github_id, token_type)
    
#     if not token_value:
#         raise HTTPException(
#             status_code=404,
#             detail=f"{token_type} token not found"
#         )
    
#     # Check if token is still valid
#     if not auth_service.is_token_valid(current_user.github_id, token_type):
#         raise HTTPException(
#             status_code=400,
#             detail=f"{token_type} token has expired"
#         )
    
#     return APIResponse(
#         success=True,
#         message=f"Retrieved {token_type} token value",
#         data={"token_type": token_type, "token_value": token_value}
#     )

