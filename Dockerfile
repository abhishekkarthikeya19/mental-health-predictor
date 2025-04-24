# Use an official Python runtime as a parent image with a more secure version
# Try Alpine for a smaller attack surface, or the latest Python version for better security
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and apply security updates
# Add specific security patches for known vulnerabilities
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Copy requirements files
COPY backend/requirements.txt /app/backend/
COPY app/requirements.txt /app/app/

# Install Python dependencies with security checks
RUN pip install --upgrade pip setuptools wheel && \
    pip install pip-audit safety && \
    # Install dependencies with pinned versions for better security
    pip install -r /app/backend/requirements.txt && \
    pip install -r /app/app/requirements.txt && \
    # Run security checks on installed packages
    pip-audit && \
    safety check && \
    # Remove security tools after use to reduce attack surface
    pip uninstall -y pip-audit safety

# Copy the rest of the application
COPY . /app/

# Train the model if it doesn't exist
RUN python -c "import os; os.makedirs('/app/app/model', exist_ok=True)" && \
    if [ ! -f /app/app/model/mental_health_model.pkl ]; then \
        python /app/app/train_model.py; \
    fi

# Create a non-root user with minimal privileges and switch to it for security
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -s /bin/false -d /app appuser && \
    chown -R appuser:appgroup /app && \
    # Set appropriate permissions
    chmod -R 755 /app

# Set security-related environment variables
ENV PYTHONHASHSEED=random \
    PYTHONWARNINGS=ignore

# Switch to non-root user
USER appuser:appgroup

# Set the working directory to the backend folder
WORKDIR /app/backend

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000", "--env", "production"]