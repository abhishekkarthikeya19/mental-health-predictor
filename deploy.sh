#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu

# Create docker-compose.yml file
cat > docker-compose.yml << EOL
version: '3'

services:
  backend:
    build: .
    image: mental-health-predictor-backend
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
sudo docker-compose up -d --build

# Install Nginx
echo "Installing Nginx..."
sudo apt-get install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Configure Nginx as a reverse proxy
sudo tee /etc/nginx/sites-available/mental-health-predictor > /dev/null << 'EOL'
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

# Enable the site and restart Nginx
sudo ln -s /etc/nginx/sites-available/mental-health-predictor /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

echo "Deployment completed successfully!"