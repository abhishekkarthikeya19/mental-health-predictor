# PowerShell script to deploy Mental Health Predictor Backend to EC2
# This script should be run on your local Windows machine

# Configuration - Replace these values with your own
$EC2_USER = "ubuntu"  # Usually "ec2-user" for Amazon Linux or "ubuntu" for Ubuntu
$EC2_HOST = "3.94.184.236"  # Your EC2 instance's public IP address
$EC2_KEY_PATH = "C:/Users/chaithanya/Downloads/mental-health-predictor-key.pem"  # Path to your new .pem key file
$PROJECT_NAME = "mental-health-predictor"
$DOCKER_IMAGE_NAME = "mental-health-predictor-backend"
$DOCKER_CONTAINER_NAME = "mental-health-predictor-backend"

# Check if key file exists
if (-not (Test-Path $EC2_KEY_PATH)) {
    Write-Error "Error: EC2 key file not found at $EC2_KEY_PATH"
    exit 1
}

# Create a temporary deployment directory
$DEPLOY_DIR = [System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString()
New-Item -ItemType Directory -Path $DEPLOY_DIR | Out-Null
Write-Host "Created temporary directory: $DEPLOY_DIR"

# Copy necessary files to the deployment directory
Write-Host "Copying project files..."
Copy-Item -Path "backend", "app", "requirements.txt", "Dockerfile", "run_app.py" -Destination $DEPLOY_DIR -Recurse

# Create a setup script to run on the EC2 instance
$setupScript = @"
#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -a -G docker ubuntu
    echo "Docker installed successfully"
else
    echo "Docker is already installed"
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed successfully"
else
    echo "Docker Compose is already installed"
fi

# Create docker-compose.yml file
cat > docker-compose.yml << EOL
version: '3'

services:
  backend:
    build: .
    image: $DOCKER_IMAGE_NAME
    container_name: $DOCKER_CONTAINER_NAME
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
if docker ps | grep -q $DOCKER_CONTAINER_NAME; then
    echo "Container is running successfully"
else
    echo "Error: Container failed to start"
    docker logs $DOCKER_CONTAINER_NAME
    exit 1
fi

# Set up Nginx as a reverse proxy (if needed)
if ! command -v nginx &> /dev/null; then
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

    # Enable the site and restart Nginx to apply the configuration
    sudo ln -s /etc/nginx/sites-available/mental-health-predictor /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo systemctl restart nginx
    echo "Nginx configured successfully"
fi

echo "Deployment completed successfully!"
"@

# Write the setup script to a file
$setupScript | Out-File -FilePath "$DEPLOY_DIR\setup_ec2.sh" -Encoding ASCII

# Make sure the setup script has Unix line endings
(Get-Content "$DEPLOY_DIR\setup_ec2.sh") | ForEach-Object { $_ -replace "`r`n", "`n" } | Set-Content "$DEPLOY_DIR\setup_ec2.sh" -NoNewline

# Compress the deployment directory
Write-Host "Compressing project files..."
$DEPLOY_ARCHIVE = "$PROJECT_NAME-deploy.zip"
Compress-Archive -Path "$DEPLOY_DIR\*" -DestinationPath $DEPLOY_ARCHIVE -Force

# Check if ssh.exe is available
$sshAvailable = $null -ne (Get-Command "ssh.exe" -ErrorAction SilentlyContinue)

if ($sshAvailable) {
    # Use built-in SSH commands if available
    Write-Host "Copying files to EC2 instance..."
    scp.exe -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no $DEPLOY_ARCHIVE "${EC2_USER}@${EC2_HOST}:~/"

    Write-Host "Setting up the EC2 instance..."
    $sshCommand = @"
mkdir -p "$PROJECT_NAME"
unzip -o "$DEPLOY_ARCHIVE" -d "$PROJECT_NAME"
cd "$PROJECT_NAME"
chmod +x setup_ec2.sh
./setup_ec2.sh
"@
    ssh.exe -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no "${EC2_USER}@${EC2_HOST}" $sshCommand
} else {
    # Inform user to install OpenSSH or use alternative methods
    Write-Host "OpenSSH client not found. Please install OpenSSH client for Windows or use WSL to run the bash script."
    Write-Host "You can install OpenSSH client by running the following command as administrator:"
    Write-Host "Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"
    exit 1
}

# Clean up
Write-Host "Cleaning up..."
Remove-Item -Path $DEPLOY_DIR -Recurse -Force
Remove-Item -Path $DEPLOY_ARCHIVE -Force

Write-Host "Deployment completed successfully!"
Write-Host "The backend API is now available at: http://$EC2_HOST:8000"
Write-Host "You may want to set up HTTPS using Let's Encrypt for production use."