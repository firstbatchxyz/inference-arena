from pydantic import BaseModel, Field, field_validator, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import re


# Enums
class PlatformEnum(str, Enum):
    OLLAMA = "Ollama"
    VLLM = "VLLM"

class GPUTypeEnum(str, Enum):
    AMD_INSTINCT_MI300X_OAM = "AMD Instinct MI300X OAM"
    NVIDIA_A100_80GB_PCIE = "NVIDIA A100 80GB PCIe"
    NVIDIA_A100_SXM4_80GB = "NVIDIA A100-SXM4-80GB"
    NVIDIA_A30 = "NVIDIA A30"
    NVIDIA_A40 = "NVIDIA A40"
    NVIDIA_B200 = "NVIDIA B200"
    NVIDIA_H100_80GB_HBM3 = "NVIDIA H100 80GB HBM3"
    NVIDIA_H100_NVL = "NVIDIA H100 NVL"
    NVIDIA_H100_PCIE = "NVIDIA H100 PCIe"
    NVIDIA_H200 = "NVIDIA H200"
    NVIDIA_L4 = "NVIDIA L4"
    NVIDIA_L40 = "NVIDIA L40"
    NVIDIA_L40S = "NVIDIA L40S"
    NVIDIA_GEFORCE_RTX_3070 = "NVIDIA GeForce RTX 3070"
    NVIDIA_GEFORCE_RTX_3080 = "NVIDIA GeForce RTX 3080"
    NVIDIA_GEFORCE_RTX_3080_TI = "NVIDIA GeForce RTX 3080 Ti"
    NVIDIA_GEFORCE_RTX_3090 = "NVIDIA GeForce RTX 3090"
    NVIDIA_GEFORCE_RTX_3090_TI = "NVIDIA GeForce RTX 3090 Ti"
    NVIDIA_GEFORCE_RTX_4070_TI = "NVIDIA GeForce RTX 4070 Ti"
    NVIDIA_GEFORCE_RTX_4080 = "NVIDIA GeForce RTX 4080"
    NVIDIA_GEFORCE_RTX_4080_SUPER = "NVIDIA GeForce RTX 4080 SUPER"
    NVIDIA_GEFORCE_RTX_4090 = "NVIDIA GeForce RTX 4090"
    NVIDIA_GEFORCE_RTX_5080 = "NVIDIA GeForce RTX 5080"
    NVIDIA_GEFORCE_RTX_5090 = "NVIDIA GeForce RTX 5090"
    NVIDIA_RTX_2000_ADA = "NVIDIA RTX 2000 Ada Generation"
    NVIDIA_RTX_4000_ADA = "NVIDIA RTX 4000 Ada Generation"
    NVIDIA_RTX_5000_ADA = "NVIDIA RTX 5000 Ada Generation"
    NVIDIA_RTX_6000_ADA = "NVIDIA RTX 6000 Ada Generation"
    NVIDIA_RTX_A2000 = "NVIDIA RTX A2000"
    NVIDIA_RTX_A4000 = "NVIDIA RTX A4000"
    NVIDIA_RTX_A4500 = "NVIDIA RTX A4500"
    NVIDIA_RTX_A5000 = "NVIDIA RTX A5000"
    NVIDIA_RTX_A6000 = "NVIDIA RTX A6000"
    NVIDIA_RTX_PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    TESLA_V100_FHHL = "Tesla V100-FHHL-16GB"
    TESLA_V100_PCIE = "Tesla V100-PCIE-16GB"
    TESLA_V100_SXM2 = "Tesla V100-SXM2-16GB"
    

# Pydantic Models
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


class BenchmarkResultInput(BaseModel):
    platform: str ## Change after to enum
    gpu_type:str ## Change after to enum
    llm_id:str ## Change after to enum
    server_type:str ## Change after to enum


# Comment Models
class CommentInput(BaseModel):
    comment: str = Field(..., min_length=1, max_length=1000, description="Comment content")
    pod_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$', description="Pod ID")
    parent_id: Optional[str] = Field(None, pattern=r'^[a-fA-F0-9]{24}$', description="Parent comment ID (MongoDB ObjectId)")
    
    @validator('comment')
    def validate_comment(cls, v):
        if not v or not v.strip():
            raise ValueError('Comment cannot be empty')
        # Basic sanitization - remove script tags and other dangerous content
        cleaned_comment = re.sub(r'<script.*?</script>', '', v, flags=re.IGNORECASE | re.DOTALL)
        cleaned_comment = re.sub(r'javascript:', '', cleaned_comment, flags=re.IGNORECASE)
        return cleaned_comment.strip()[:1000]  # Ensure max length
    
    @validator('pod_id')
    def validate_pod_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Pod ID contains invalid characters')
        return v
    
    @validator('parent_id')
    def validate_parent_id(cls, v):
        if v is not None and not re.match(r'^[a-fA-F0-9]{24}$', v):
            raise ValueError('Parent ID must be a valid MongoDB ObjectId')
        return v

class CommentResponse(BaseModel):
    id: str = Field(..., alias="_id")
    comment: str
    likes: int
    created_at: datetime
    pod_id: str
    user_id: int
    username: str
    user_email: str
    user_avatar: Optional[str] = None
    user_has_liked: bool = False
    parent_id: Optional[str] = None
    is_parent: bool = True
    has_replies: bool = False
    reply_count: int = 0
    ranking_score: float = 0.0
    depth: int = 0  # 0 for top-level, 1 for replies
    replies: List['CommentResponse'] = []  # Child comments for hierarchical structure


class LikeInput(BaseModel):
    comment_id: str

class PodUpvoteInput(BaseModel):
    pod_id: str

class ChatMessageInput(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Chat message content")
    pod_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$', description="Pod ID")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        # Remove potentially dangerous characters
        cleaned_message = re.sub(r'[<>"\'\&]', '', v.strip())
        if len(cleaned_message) < 1:
            raise ValueError('Message must contain valid characters')
        return cleaned_message[:2000]  # Ensure max length
    
    @validator('pod_id')
    def validate_pod_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Pod ID contains invalid characters')
        return v

class ChatLogResponse(BaseModel):
    id: str = Field(..., alias="_id")
    user_message: str
    ai_response: str
    timestamp: datetime
    pod_id: str
    user_id: int
    username: str
    user_email: str
    user_avatar: Optional[str] = None
    is_authenticated: bool = True

class ChatLogDeleteInput(BaseModel):
    log_id: str


# User Models
class User(BaseModel):
    github_id: int
    username: str
    email: str
    avatar_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime = Field(default_factory=datetime.utcnow)


class UserTokens(BaseModel):
    github_id: int
    github_access_token: Optional[str] = None
    github_refresh_token: Optional[str] = None
    github_token_expires_at: Optional[datetime] = None
    runpod_api_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    other_tokens: Optional[Dict[str, Any]] = None  # For any other tokens
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TokenUpdateRequest(BaseModel):
    token_type: str  # "github", "runpod", "openai", "anthropic", etc.
    token_value: str
    expires_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    token_type: str
    has_token: bool
    expires_at: Optional[datetime] = None


class GitHubOAuthResponse(BaseModel):
    access_token: str
    token_type: str
    scope: str


class GitHubUserInfo(BaseModel):
    id: int
    login: str
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    name: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: User


class GithubLoginResponse(BaseModel):
    auth_url: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "auth_url": "https://github.com/login/oauth/authorize?client_id=1234567890&redirect_uri=http://localhost:8000/auth/github/callback&scope=user:email,read:user"
            }
        }
    }

class GithubCallbackResponse(BaseModel):
    access_token: str
    token_type: str
    user: User

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "1234567890",
                "token_type": "bearer",
                "user": {
                "github_id": 86774109,
                "username": "fatihbugrakdogan",
                "email": "fatihbugrakdogan@gmail.com",
                "avatar_url": "https://avatars.githubusercontent.com/u/86774109?v=4",
                "created_at": "2025-07-08T21:52:44.749000",
                "last_login": "2025-07-14T15:45:11.393000"                }
                }
        }
    }

class AuthCurrentUserResponse(BaseModel):
    user: User

    model_config = {
        "json_schema_extra": {
            "example": {
                "user": {
                    "github_id": 86774109,
                    "username": "fatihbugrakdogan",
                    "email": "fatihbugrakdogan@gmail.com",
                    "avatar_url": "https://avatars.githubusercontent.com/u/86774109?v=4",
                    "created_at": "2025-07-08T21:52:44.749000",
                    "last_login": "2025-07-14T15:45:40.372000"
                }
            }
        }
    }


class AuthLogoutResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "Successfully logged out. Please discard your access token."
            }
        }
    }


class AuthRefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: User

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "1234567890",
                "token_type": "bearer",
                "user": {
                    "github_id": 86774109,
                    "username": "fatihbugrakdogan",
                    "email": "fatihbugrakdogan@gmail.com",
                    "avatar_url": "https://avatars.githubusercontent.com/u/86774109?v=4",
                    "created_at": "2025-07-08T21:52:44.749000",
                    "last_login": "2025-07-14T15:45:40.372000"
                }
            }
        }
    }

class BenchmarkResult(BaseModel):
    pod_id: str
    model_name: str
    benchmark_type: str
    parameter_size: str
    inference_engine: str
    hardware: str
    memory: Optional[Union[str, int]]  # Can be None or numeric
    download_speed: Optional[str]  # Can be None
    server_type: str
    cost: Optional[float]  # Should be float
    avg_ttft_ms: Optional[float]  # Should be float
    avg_throughput: Optional[float]  # Should be float
    test_duration: Optional[float]  # Should be float
    container_setup: Optional[Union[str, float]]  # Can be None or numeric
    avg_input_token_per_request: Optional[float]  # Should be float
    avg_output_token_per_request: Optional[float]  # Should be float
    total_requests: Optional[int]  # Should be int
    model_download_time: Optional[int]  # Can be None or numeric
    server_cost_per_hour: Optional[float]  # Should be float
    graph_info: List[dict] = []
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    likes: int = 0
    comment_count: int = 0
    is_upvoted_by_user: bool = False
    icon_url: Optional[str] = None
    # liked_user_list: List[int] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "pod_id": "1234567890",
                "model_name": "llama3.1",
                "benchmark_type": "throughput",
                "parameter_size": "8B",
                "inference_engine": "ollama",
                "hardware": "NVIDIA A100 80GB PCIe",
                "memory": "8GB",
                "download_speed": "100MB/s",
                "server_type": "ollama",
                "cost": 100.0,
                "avg_ttft_ms": 100.0,
                "avg_throughput": 100.0,
                "test_duration": 100.0,
                "container_setup": "100",
                "avg_input_token_per_request": 100.0,
                "avg_output_token_per_request": 100.0,
                "total_requests": 100,
                "model_download_time": "100",
                "server_cost_per_hour": 100.0,
                "graph_info": [{"x": "100", "y": "100"}, {"x": "200", "y": "200"}],
                "x_axis_label": "Rate",
                "y_axis_label": "Request Latency",
                "likes": 100,
                "liked_user_list": [86774109]
            }
        }
    }

class BenchmarkResultQuery(BaseModel):
    pod_id:str
    model_name:str
    benchmark_type:str


class PaginationMeta(BaseModel):
    current_page: int
    per_page: int
    total_items: int
    total_pages: int
    has_next: bool
    has_prev: bool

class BenchmarkResultList(BaseModel):
    benchmark_results: List[BenchmarkResult]
    pagination: Optional[PaginationMeta] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "benchmark_results": [
                    {
                        "pod_id": "1234567890",
                        "model_name": "llama3.1",
                        "benchmark_type": "throughput",
                        "parameter_size": "8B",
                        "inference_engine": "ollama",
                        "hardware": "NVIDIA A100 80GB PCIe",
                        "memory": "8GB",
                        "download_speed": "100MB/s",
                        "server_type": "ollama",
                        "cost": "100",
                        "avg_ttft_ms": "100",
                        "avg_throughput": "100",
                        "test_duration": "100",
                        "container_setup": "100",
                        "avg_input_token_per_request": "100",
                        "avg_output_token_per_request": "100",
                        "total_requests": "100",
                        "model_download_time": "100",
                        "server_cost_per_hour": "100",
                    },
                    
                ],
                "pagination": {
                    "current_page": 1,
                    "per_page": 6,
                    "total_items": 20,
                    "total_pages": 4,
                    "has_next": True,
                    "has_prev": False
                }
            }
        }
    }


class BenchmarkResultFilterInput(BaseModel):
    model_name: Optional[str] = None
    inference_engine: Optional[str] = None
    parameter_size: Optional[str] = None
    hardware: Optional[str] = None
    hardware_price: Optional[str] = None
    TTFT: Optional[str] = None
    TPS: Optional[str] = None
    benchmark_type: Optional[str] = None
    hourly_cost: Optional[str] = None


class BenchmarkResultFilter(BaseModel):
    pod_id: str
    model_name: str
    parameter_size: str
    inference_engine: str
    hardware: str
    likes: Optional[int] = 0
    comment_count: Optional[int] = 0
    icon_url: Optional[str] = None
    # liked_user_list:List[int]

class BenchmarkResultFilterResponse(BaseModel):
    benchmark_results: List[BenchmarkResultFilter]
    pagination: Optional[PaginationMeta] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "benchmark_results": [
                    {
                    "pod_id":"1234567890",
                    "model_name":"llama3.1",
                    "parameter_size":"8B",
                    "inference_engine":"ollama",
                    "hardware":"NVIDIA A100 80GB PCIe",
                    "likes":100,
                    "liked_user_list":[86774109]
                    }
                ],
                "pagination": {
                    "current_page": 1,
                    "per_page": 6,
                    "total_items": 15,
                    "total_pages": 3,
                    "has_next": True,
                    "has_prev": False
                }
            }
        }
    }



class ChatResponse(BaseModel):
    response:str
    timestamp:datetime
    log_id:str
    pod_id:str
    user_authenticated:bool
    user_info:str

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "Hello, how are you?",
                "timestamp": "2025-07-14T15:45:40.372000",
                "log_id": "1234567890",
                "pod_id": "1234567890",
                "user_authenticated": True,
                "user_info": {
                    "user_id": 86774109,
                    "username": "fatihbugrakdogan"
                }
            }
        }
    }

class PostCommentResponse(BaseModel):
    comment_id:str
    pod_id:str
    comment:str
    user_id:int
    username:str
    parent_id:str
    depth:int

    model_config = {
        "json_schema_extra": {
            "example": {
                "comment_id": "1234567890",
                "pod_id": "1234567890",
                "comment": "Hello, how are you?",
                "user_id": 86774109,
                "username": "fatihbugrakdogan",
                "parent_id": "1234567890",
                "depth": 0
            }
        }
    }

class GetCommentsResponse(BaseModel):
    parent_comments:List[CommentResponse]
    total_comments:int
    total_parents:int
    total_replies:int

    model_config = {
        "json_schema_extra": {
            "example": {
                "parent_comments": [
                    {
                        "comment_id": "1234567890",
                        "pod_id": "1234567890",
                        "comment": "Hello, how are you?",
                        "user_id": 86774109,
                        "username": "fatihbugrakdogan",
                        "parent_id": "1234567890",
                        "depth": 0
                    }
                ],
                "total_comments": 1,
                "total_parents": 1,
                "total_replies": 0
            }
        }
    }


class GetAllCommentsResponse(BaseModel):
    comments:List[CommentResponse]

    model_config = {
        "json_schema_extra": {
            "example": {
                "comments": [
                    {
                        "comment_id": "1234567890",
                        "pod_id": "1234567890",
                        "comment": "Hello, how are you?",
                        "user_id": 86774109,
                        "username": "fatihbugrakdogan",
                        "parent_id": "1234567890",
                        "depth": 0
                    }
                ]
            }
        }
    }   


class LikeResponse(BaseModel):
    comment_id:str
    user_id:int
    username:str
    total_likes:int
    ranking_score:float

    model_config = {
        "json_schema_extra": {
            "example": {
                "comment_id": "1234567890",
                "user_id": 86774109,
                "username": "fatihbugrakdogan",
                "total_likes": 1,
                "ranking_score": 1.0
            }
        }
    }

class RemoveLikeResponse(BaseModel):
    comment_id:str
    user_id:int
    username:str
    total_likes:int
    ranking_score:float

    model_config = {
        "json_schema_extra": {
            "example": {
                "comment_id": "1234567890",
                "user_id": 86774109,
                "username": "fatihbugrakdogan",
                "total_likes": 1,
                "ranking_score": 1.0
            }
        }
    }

class DeleteCommentResponse(BaseModel):
    comment_id:str

    model_config = {
        "json_schema_extra": {
            "example": {
                "comment_id": "1234567890"
            }
        }
    }

class GetUserCommentsResponse(BaseModel):
    comments:List[CommentResponse]

    model_config = {
        "json_schema_extra": {
            "example": {
                "comments": [
                    {
                        "comment_id": "1234567890",
                        "pod_id": "1234567890",
                        "comment": "Hello, how are you?",
                        "user_id": 86774109,
                        "username": "fatihbugrakdogan",
                        "parent_id": "1234567890",
                        "depth": 0
                    }
                ]
            }
        }
    }

class GetChatLogsResponse(BaseModel):
    logs:List[ChatLogResponse]

    model_config = {
        "json_schema_extra": {
            "example": {
                "logs": [
                    {
                        "log_id": "1234567890", 
                        "user_message": "Hello, how are you?",
                        "ai_response": "I'm good, thank you!",
                        "timestamp": "2025-07-14T15:45:40.372000",
                        "pod_id": "1234567890",
                        "user_id": 86774109,
                        "username": "fatihbugrakdogan",
                        "user_email": "fatihbugrakdogan@gmail.com",
                        "user_avatar": "https://avatars.githubusercontent.com/u/86774109?v=4"
                    }
                ]
            }
        }
    }

class GetUserChatLogsResponse(BaseModel):
    logs:List[ChatLogResponse]

    model_config = {
        "json_schema_extra": {
            "example": {
                "logs": [
                    {
                        "log_id": "1234567890",
                        "user_message": "Hello, how are you?",
                        "ai_response": "I'm good, thank you!",
                        "timestamp": "2025-07-14T15:45:40.372000",
                        "pod_id": "1234567890",
                        "user_id": 86774109,
                        "username": "fatihbugrakdogan",
                        "user_email": "fatihbugrakdogan@gmail.com",
                        "user_avatar": "https://avatars.githubusercontent.com/u/86774109?v=4"   
                    }
                ]
            }
        }
    }


class GetPodChatLogsResponse(BaseModel):
    logs:List[ChatLogResponse]

    model_config = {
        "json_schema_extra": {
            "example": {
                "logs": [
                    {
                        "log_id": "1234567890",
                        "user_message": "Hello, how are you?",
                        "ai_response": "I'm good, thank you!",
                        "timestamp": "2025-07-14T15:45:40.372000",
                        "pod_id": "1234567890",
                        "user_id": 86774109,
                        "username": "fatihbugrakdogan",
                        "user_email": "fatihbugrakdogan@gmail.com",
                        "user_avatar": "https://avatars.githubusercontent.com/u/86774109?v=4"
                    }
                ]
            }
        }
    }


class DeleteChatLogResponse(BaseModel):
    log_id:str

    model_config = {
        "json_schema_extra": {
            "example": {
                "log_id": "1234567890"
            }
        }
    }

class ContentModerationError(BaseModel):
    error_type: str
    message: str
    field: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "error_type": "CONTENT_MODERATION_FAILED",
                "message": "Your comment contains inappropriate content. Please revise your comment to be more professional and respectful.",
                "field": "comment"
            }
        }
    }

class GpuPrice(BaseModel):
    platform:str
    name:str
    price:float

    model_config = {
        "json_schema_extra": {
            "example": {
                "platform": "runpod",
                "name": "NVIDIA A100 80GB PCIe",
                "price": 100
            }
        }
    }



class GetGpuPricesByPlatformResponse(BaseModel):
    gpu_prices:List[GpuPrice]

    model_config = {
        "json_schema_extra": {
            "example": {
                "gpu_prices": [
                    {
                        "platform": "runpod",
                        "name": "NVIDIA A100 80GB PCIe",
                        "price": 100,
                    }
                ]
            }
        }
    }


class GetAvailableBenchmarkOptionsResponse(BaseModel):
    model_names:List[str]
    inference_engines:List[str]
    parameter_sizes:List[str]
    hardwares:List[str]
    server_types:List[str]

