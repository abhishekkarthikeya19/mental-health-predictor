#!/bin/bash

# Deploy Mental Health Predictor Backend to EC2
# This script should be run on your local machine

# Configuration - Replace these values with your own
EC2_USER="ec2-user"
EC2_HOST="your-ec2-instance-ip"
EC2_KEY_PATH="path/to/your-key.pem"
PROJECT_NAME="mental-health-predictor"
DOCKER_IMAGE_NAME="mental-health-predictor-backend"
DOCKER_CONTAINER_NAME="mental-health-predictor-backend"

# Check if key file exists
if [ ! -f "$EC2_KEY_PATH" ]; then
    echo "Error: EC2 key file not found at $EC2_KEY_PATH"
    exit 1
fi

# Create a temporary deployment directory
DEPLOY_DIR=$(mktemp -d)
echo "Created temporary directory: $DEPLOY_DIR"

# Copy necessary files to the deployment directory
echo "Copying project files..."
cp -r backend app requirements.txt Dockerfile run_app.py "$DEPLOY_DIR"

# Create a setup script to run on the EC2 instance
cat > "$DEPLOY_DIR/setup_ec2.sh" << 'EOF'
#!/bin/bash

# Update system packages
sudo yum update -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo amazon-linux-extras install docker -y
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -a -G docker ec2-user
    echo "Docker installed successfully"
else
    echo "Docker is already installed"
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed successfully"
else
    echo "Docker Compose is already installed"
fi

# Create docker-compose.yml file
cat > docker-compose.yml << 'EOL'
version: '3'

services:
  backend:
    build: .
    container_name: mental-health-predictor-backend
    ports:
      - "8000:8000"
    restart: always
    environment:
      - ENVIRONMENT=production
      - CORS_ORIGINS=https://chaithanya.github.io
      - RATE_LIMIT_REQUESTS=100
      - RATE_LIMIT_PERIOD=3600
    volumes:
      - ./app/model:/app/app/model
EOL

# Build and start the Docker container
echo "Building and starting the Docker container..."
docker-compose up -d --build

# Check if the container is running
if docker ps | grep -q mental-health-predictor-backend; then
    echo "Container is running successfully"
else
    echo "Error: Container failed to start"
    docker logs mental-health-predictor-backend
    exit 1
fi

# Set up Nginx as a reverse proxy (if needed)
if ! command -v nginx &> /dev/null; then
    echo "Installing Nginx..."
    sudo amazon-linux-extras install nginx1 -y
    sudo systemctl start nginx
    sudo systemctl enable nginx
    
    # Configure Nginx as a reverse proxy
    sudo tee /etc/nginx/conf.d/mental-health-predictor.conf > /dev/null << 'EOL'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOL

    # Restart Nginx to apply the configuration
    sudo systemctl restart nginx
    echo "Nginx configured successfully"
fi

echo "Deployment completed successfully!"
EOF

# Make the setup script executable
chmod +x "$DEPLOY_DIR/setup_ec2.sh"

# Compress the deployment directory
echo "Compressing project files..."
DEPLOY_ARCHIVE="$PROJECT_NAME-deploy.tar.gz"
tar -czf "$DEPLOY_ARCHIVE" -C "$DEPLOY_DIR" .

# Copy the archive to the EC2 instance
echo "Copying files to EC2 instance..."
scp -i "$EC2_KEY_PATH" "$DEPLOY_ARCHIVE" "$EC2_USER@$EC2_HOST:~/"

# SSH into the EC2 instance and run the setup script
echo "Setting up the EC2 instance..."
ssh -i "$EC2_KEY_PATH" "$EC2_USER@$EC2_HOST" << EOF
mkdir -p "$PROJECT_NAME"
tar -xzf "$DEPLOY_ARCHIVE" -C "$PROJECT_NAME"
cd "$PROJECT_NAME"
chmod +x setup_ec2.sh
./setup_ec2.sh
EOF

# Clean up
echo "Cleaning up..."
rm -rf "$DEPLOY_DIR"
rm "$DEPLOY_ARCHIVE"

echo "Deployment completed successfully!"
echo "The backend API is now available at: http://$EC2_HOST:8000"
echo "You may want to set up HTTPS using Let's Encrypt for production use."