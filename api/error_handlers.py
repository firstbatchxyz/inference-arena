from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from slowapi.errors import RateLimitExceeded
import logging
import traceback
from typing import Any, Dict
import json

logger = logging.getLogger(__name__)

class APIError(HTTPException):
    """Custom API error with additional context."""
    
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ):
        self.message = message
        self.error_code = error_code or f"API_ERROR_{status_code}"
        self.details = details or {}
        
        detail = {
            "error_code": self.error_code,
            "message": message,
            "details": self.details
        }
        
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class SecurityError(APIError):
    """Security-related error."""
    
    def __init__(self, message: str = "Security violation detected", details: Dict[str, Any] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            message=message,
            error_code="SECURITY_ERROR",
            details=details
        )

class ResourceNotFoundError(APIError):
    """Resource not found error."""
    
    def __init__(self, resource: str, identifier: str = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details={"resource": resource, "identifier": identifier}
        )

class ValidationError(APIError):
    """Input validation error."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )

class DatabaseError(APIError):
    """Database operation error."""
    
    def __init__(self, message: str = "Database operation failed", operation: str = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_code="DATABASE_ERROR",
            details={"operation": operation} if operation else {}
        )

class ExternalServiceError(APIError):
    """External service integration error."""
    
    def __init__(self, service: str, message: str = None, status_code: int = None):
        message = message or f"{service} service is unavailable"
        status_code = status_code or status.HTTP_503_SERVICE_UNAVAILABLE
        
        super().__init__(
            status_code=status_code,
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service}
        )

def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: Dict[str, Any] = None,
    request_id: str = None
) -> JSONResponse:
    """Create standardized error response."""
    
    response_data = {
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": logger.handlers[0].formatter.formatTime(logging.LogRecord(
                name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
            )) if hasattr(logger.handlers[0], 'formatter') else None
        }
    }
    
    if details:
        response_data["error"]["details"] = details
    
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )

async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors."""
    
    # Log the error with context
    logger.error(
        f"API Error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "endpoint": str(request.url),
            "method": request.method,
            "details": exc.details
        }
    )
    
    return create_error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details
    )

async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    
    # Extract field names and error messages
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(x) for x in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        f"Validation error on {request.method} {request.url}",
        extra={
            "endpoint": str(request.url),
            "method": request.method,
            "validation_errors": errors
        }
    )
    
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="VALIDATION_ERROR",
        message="Input validation failed",
        details={"validation_errors": errors}
    )

async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors."""
    
    logger.warning(
        f"Rate limit exceeded for {request.client.host}",
        extra={
            "endpoint": str(request.url),
            "method": request.method,
            "client_ip": request.client.host,
            "rate_limit": str(exc.detail)
        }
    )
    
    return create_error_response(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        error_code="RATE_LIMIT_EXCEEDED",
        message="Too many requests. Please try again later.",
        details={"retry_after": "60 seconds"}
    )

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard HTTP exceptions."""
    
    # Don't log 404s and other client errors as errors
    if 400 <= exc.status_code < 500:
        log_level = logging.WARNING
    else:
        log_level = logging.ERROR
    
    logger.log(
        log_level,
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "endpoint": str(request.url),
            "method": request.method,
            "detail": exc.detail
        }
    )
    
    # Handle structured detail (dict) vs simple string detail
    if isinstance(exc.detail, dict):
        return create_error_response(
            status_code=exc.status_code,
            error_code=exc.detail.get("error_code", f"HTTP_{exc.status_code}"),
            message=exc.detail.get("message", "An error occurred"),
            details=exc.detail.get("details")
        )
    else:
        return create_error_response(
            status_code=exc.status_code,
            error_code=f"HTTP_{exc.status_code}",
            message=str(exc.detail)
        )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    
    # Log full traceback for debugging
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "endpoint": str(request.url),
            "method": request.method,
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }
    )
    
    # Don't expose internal error details in production
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_SERVER_ERROR",
        message="An internal server error occurred"
    ) 