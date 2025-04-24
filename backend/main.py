# main.py
"""
Main FastAPI application for the Mental Health Predictor API.
"""
import logging
import sys
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import models and utilities
try:
    # When imported as a module (e.g., from backend.main import app)
    from .models import PredictionRequest, PredictionResponse, ErrorResponse, HealthCheckResponse
    from .utils.patching import apply_anyio_patch, apply_logging_patch
    from .utils.model_loader import load_model, ModelNotFoundError, InvalidModelError
    from .middleware import RateLimitMiddleware, LoggingMiddleware
    from .config import settings
except ImportError:
    # When run directly (e.g., python main.py)
    from models import PredictionRequest, PredictionResponse, ErrorResponse, HealthCheckResponse
    from utils.patching import apply_anyio_patch, apply_logging_patch
    from utils.model_loader import load_model, ModelNotFoundError, InvalidModelError
    from middleware import RateLimitMiddleware, LoggingMiddleware
    from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Apply patches
apply_anyio_patch()
apply_logging_patch()

# Initialize FastAPI with metadata for documentation
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(
    RateLimitMiddleware,
    requests_limit=settings.RATE_LIMIT_REQUESTS,
    period=settings.RATE_LIMIT_PERIOD
)
app.add_middleware(LoggingMiddleware)

# Load the model
try:
    model = load_model(settings.MODEL_PATHS)
    logger.info("Model loaded successfully")
except (ModelNotFoundError, InvalidModelError) as e:
    logger.critical(f"Failed to load model: {str(e)}")
    sys.exit(1)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {str(exc.detail)}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# Define the root route
@app.get("/", response_model=dict)
def read_root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the Mental Health Predictor API!"}

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "healthy",
        "version": settings.API_VERSION
    }

# Define the prediction endpoint with improved error handling and response
@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Analyze text for signs of mental distress.
    
    Args:
        request: The prediction request containing the text to analyze
        
    Returns:
        PredictionResponse: The prediction result with confidence score and recommendation
        
    Raises:
        HTTPException: If an error occurs during prediction
    """
    try:
        # Get prediction
        text = request.text_input
        prediction = int(model.predict([text])[0])
        
        # Get confidence score (probability)
        confidence = float(model.predict_proba([text])[0][prediction])
        
        # Generate recommendation based on prediction
        if prediction == 1:
            recommendation = settings.DISTRESS_RECOMMENDATION
        else:
            recommendation = settings.NORMAL_RECOMMENDATION
        
        # Log the prediction (without the full text for privacy)
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "recommendation": recommendation
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Keep the old endpoint for backward compatibility
@app.post("/predict/legacy", response_model=dict)
async def predict_legacy(request: Request):
    """
    Legacy endpoint for backward compatibility.
    
    Args:
        request: The HTTP request containing the text to analyze
        
    Returns:
        dict: A dictionary containing just the prediction result
        
    Raises:
        HTTPException: If an error occurs during prediction or if the input is invalid
    """
    try:
        # Parse the request body manually
        data = await request.json()
        text = data.get("text_input", "")
        
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Missing or empty text_input field"
            )
        
        # Get prediction
        prediction = int(model.predict([text])[0])
        
        # Log the prediction (without the full text for privacy)
        logger.info(f"Legacy prediction: {prediction}")
        
        # Return just the prediction for backward compatibility
        return {"prediction": prediction}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in legacy prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

