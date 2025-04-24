# middleware.py
"""
Custom middleware for the FastAPI application.
"""
import time
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from typing import Dict, List, Callable

try:
    # When imported as a module
    from .config import settings
except ImportError:
    # When run directly
    from config import settings

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.
    """
    def __init__(self, app, requests_limit: int = 100, period: int = 3600):
        """
        Initialize the rate limiter.
        
        Args:
            app: The FastAPI application
            requests_limit: Maximum number of requests allowed in the period
            period: Time period in seconds
        """
        super().__init__(app)
        self.requests_limit = requests_limit
        self.period = period
        self.request_counts: Dict[str, List[float]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and apply rate limiting.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response: The response from the next middleware or route handler
        """
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for certain paths
        if request.url.path in ["/", "/docs", "/redoc", "/openapi.json", "/health"]:
            return await call_next(request)
        
        # Check if client has exceeded rate limit
        current_time = time.time()
        
        # Initialize request count for new clients
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Remove requests older than the period
        self.request_counts[client_ip] = [
            timestamp for timestamp in self.request_counts[client_ip]
            if current_time - timestamp < self.period
        ]
        
        # Check if client has exceeded rate limit
        if len(self.request_counts[client_ip]) >= self.requests_limit:
            logger.warning(f"Rate limit exceeded for client: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
        
        # Add current request timestamp
        self.request_counts[client_ip].append(current_time)
        
        # Process the request
        return await call_next(request)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log information.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response: The response from the next middleware or route handler
        """
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(f"Response: {response.status_code} (took {process_time:.4f}s)")
            
            return response
        except Exception as e:
            # Log exception
            process_time = time.time() - start_time
            logger.error(f"Exception: {str(e)} (took {process_time:.4f}s)")
            raise