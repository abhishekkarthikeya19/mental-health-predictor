#!/bin/bash

# Manual deployment of Mental Health Predictor Backend to EC2
# This script should be run on your local machine

# Configuration - Replace these values with your own
EC2_USER="ec2-user"
EC2_HOST="your-ec2-instance-ip"
EC2_KEY_PATH="path/to/your-key.pem"
PROJECT_NAME="mental-health-predictor"

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
cp -r backend app requirements.txt run_app.py "$DEPLOY_DIR"

# Create a setup script to run on the EC2 instance
cat > "$DEPLOY_DIR/setup_ec2_manual.sh" << 'EOF'
#!/bin/bash

# Update system packages
sudo yum update -y

# Install Python 3 and development tools
sudo yum install -y python3 python3-devel python3-pip gcc git

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create a systemd service file for the application
sudo tee /etc/systemd/system/mental-health-predictor.service > /dev/null << 'EOL'
[Unit]
Description=Mental Health Predictor API
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/mental-health-predictor
ExecStart=/home/ec2-user/mental-health-predictor/venv/bin/python run_app.py
Restart=always
Environment=ENVIRONMENT=production
Environment=CORS_ORIGINS=https://chaithanya.github.io
Environment=RATE_LIMIT_REQUESTS=100
Environment=RATE_LIMIT_PERIOD=3600

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable mental-health-predictor
sudo systemctl start mental-health-predictor

# Check the service status
sudo systemctl status mental-health-predictor

# Set up Nginx as a reverse proxy
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

echo "Deployment completed successfully!"
EOF

# Make the setup script executable
chmod +x "$DEPLOY_DIR/setup_ec2_manual.sh"

# Compress the deployment directory
echo "Compressing project files..."
DEPLOY_ARCHIVE="$PROJECT_NAME-deploy-manual.tar.gz"
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
chmod +x setup_ec2_manual.sh
./setup_ec2_manual.sh
EOF

# Clean up
echo "Cleaning up..."
rm -rf "$DEPLOY_DIR"
rm "$DEPLOY_ARCHIVE"

echo "Deployment completed successfully!"
echo "The backend API is now available at: http://$EC2_HOST:8000"
echo "You may want to set up HTTPS using Let's Encrypt for production use."