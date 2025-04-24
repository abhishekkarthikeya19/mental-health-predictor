# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY backend/requirements.txt /app/backend/
COPY app/requirements.txt /app/app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir -r /app/app/requirements.txt

# Copy the rest of the application
COPY . /app/

# Train the model if it doesn't exist
RUN python -c "import os; os.makedirs('/app/app/model', exist_ok=True)" && \
    if [ ! -f /app/app/model/mental_health_model.pkl ]; then \
        python /app/app/train_model.py; \
    fi

# Set the working directory to the backend folder
WORKDIR /app/backend

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "backend.run", "--host", "0.0.0.0", "--port", "8000", "--env", "production"]