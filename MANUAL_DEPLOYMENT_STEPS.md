# Manual Deployment Steps

Follow these steps to manually deploy your Mental Health Predictor backend to your EC2 instance.

## Step 1: Connect to Your EC2 Instance

```bash
ssh -i "C:/Users/chaithanya/Downloads/mental-health-predictor-key.pem" ubuntu@3.94.184.236
```

## Step 2: Create a Directory for Your Project

```bash
mkdir -p mental-health-predictor
cd mental-health-predictor
```

## Step 3: Copy the Deployment Script

Create a file named `deploy.sh` with the following content:

```bash
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
```

## Step 4: Make the Script Executable

```bash
chmod +x deploy.sh
```

## Step 5: Copy Your Application Files to the EC2 Instance

From your local machine, run:

```powershell
# Run this on your local Windows machine, not on the EC2 instance
cd C:\Users\chaithanya\Downloads\mental-health-predictor
scp -i "C:/Users/chaithanya/Downloads/mental-health-predictor-key.pem" -r backend app requirements.txt Dockerfile run_app.py ubuntu@3.94.184.236:~/mental-health-predictor/
```

## Step 6: Run the Deployment Script

Back on your EC2 instance:

```bash
./deploy.sh
```

## Step 7: Verify the Deployment

Once the script completes, your backend API should be available at:

```
http://3.94.184.236:8000
```

You can test it by accessing this URL in your browser or using tools like Postman.

## Troubleshooting

If you encounter any issues:

1. Check Docker container logs:
   ```bash
   docker logs mental-health-predictor-backend
   ```

2. Check Nginx logs:
   ```bash
   sudo tail -f /var/log/nginx/error.log
   ```

3. Check if the Docker container is running:
   ```bash
   docker ps
   ```

4. Restart the Docker container:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

5. Restart Nginx:
   ```bash
   sudo systemctl restart nginx
   ```