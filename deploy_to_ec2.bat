@echo off
echo ===== Mental Health Predictor Backend EC2 Deployment =====
echo.
echo This script will help you deploy the backend to an Amazon EC2 instance.
echo.

set /p EC2_HOST=Enter your EC2 instance IP address: 
set /p EC2_USER=Enter your EC2 username (default: ec2-user): 
if "%EC2_USER%"=="" set EC2_USER=ec2-user
set /p EC2_KEY_PATH=Enter the path to your EC2 key file (.pem): 

echo.
echo Step 1: Creating deployment package...
echo.

if not exist "temp" mkdir temp
xcopy /E /I /Y backend temp\backend
xcopy /E /I /Y app temp\app
copy requirements.txt temp\
copy Dockerfile temp\
copy run_app.py temp\

echo Creating setup script...
(
echo #!/bin/bash
echo.
echo # Update system packages
echo sudo yum update -y
echo.
echo # Install Docker if not already installed
echo if ! command -v docker ^&^> /dev/null; then
echo     echo "Installing Docker..."
echo     sudo amazon-linux-extras install docker -y
echo     sudo systemctl start docker
echo     sudo systemctl enable docker
echo     sudo usermod -a -G docker %EC2_USER%
echo     echo "Docker installed successfully"
echo else
echo     echo "Docker is already installed"
echo fi
echo.
echo # Install Docker Compose if not already installed
echo if ! command -v docker-compose ^&^> /dev/null; then
echo     echo "Installing Docker Compose..."
echo     sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
echo     sudo chmod +x /usr/local/bin/docker-compose
echo     echo "Docker Compose installed successfully"
echo else
echo     echo "Docker Compose is already installed"
echo fi
echo.
echo # Create docker-compose.yml file
echo cat ^> docker-compose.yml ^<^< 'EOL'
echo version: '3'
echo.
echo services:
echo   backend:
echo     build: .
echo     container_name: mental-health-predictor-backend
echo     ports:
echo       - "8000:8000"
echo     restart: always
echo     environment:
echo       - ENVIRONMENT=production
echo       - CORS_ORIGINS=https://chaithanya.github.io
echo       - RATE_LIMIT_REQUESTS=100
echo       - RATE_LIMIT_PERIOD=3600
echo     volumes:
echo       - ./app/model:/app/app/model
echo EOL
echo.
echo # Build and start the Docker container
echo echo "Building and starting the Docker container..."
echo docker-compose up -d --build
echo.
echo # Check if the container is running
echo if docker ps ^| grep -q mental-health-predictor-backend; then
echo     echo "Container is running successfully"
echo else
echo     echo "Error: Container failed to start"
echo     docker logs mental-health-predictor-backend
echo     exit 1
echo fi
echo.
echo # Set up Nginx as a reverse proxy (if needed)
echo if ! command -v nginx ^&^> /dev/null; then
echo     echo "Installing Nginx..."
echo     sudo amazon-linux-extras install nginx1 -y
echo     sudo systemctl start nginx
echo     sudo systemctl enable nginx
echo     
echo     # Configure Nginx as a reverse proxy
echo     sudo tee /etc/nginx/conf.d/mental-health-predictor.conf ^> /dev/null ^<^< 'EOL'
echo server {
echo     listen 80;
echo     server_name _;
echo.
echo     location / {
echo         proxy_pass http://localhost:8000;
echo         proxy_set_header Host $host;
echo         proxy_set_header X-Real-IP $remote_addr;
echo         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
echo         proxy_set_header X-Forwarded-Proto $scheme;
echo     }
echo }
echo EOL
echo.
echo     # Restart Nginx to apply the configuration
echo     sudo systemctl restart nginx
echo     echo "Nginx configured successfully"
echo fi
echo.
echo echo "Deployment completed successfully!"
) > temp\setup_ec2.sh

echo.
echo Step 2: Copying files to EC2 instance...
echo.

cd temp
tar -czf ../ec2-deploy.tar.gz *
cd ..

echo Using SSH to copy files to EC2...
ssh -i "%EC2_KEY_PATH%" %EC2_USER%@%EC2_HOST% "mkdir -p mental-health-predictor"
scp -i "%EC2_KEY_PATH%" ec2-deploy.tar.gz %EC2_USER%@%EC2_HOST%:~/mental-health-predictor/

echo.
echo Step 3: Setting up the EC2 instance...
echo.

ssh -i "%EC2_KEY_PATH%" %EC2_USER%@%EC2_HOST% "cd mental-health-predictor && tar -xzf ec2-deploy.tar.gz && chmod +x setup_ec2.sh && ./setup_ec2.sh"

echo.
echo Step 4: Cleaning up...
echo.

rmdir /S /Q temp
del ec2-deploy.tar.gz

echo.
echo ===== Deployment Complete =====
echo.
echo Your backend has been deployed to EC2.
echo It may take a few minutes for the server to start.
echo.
echo Please visit: http://%EC2_HOST%:8000
echo.
pause