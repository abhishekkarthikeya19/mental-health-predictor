#!/bin/bash

# Check the status of the Mental Health Predictor Backend on EC2
# This script should be run on your local machine

# Configuration - Replace these values with your own
EC2_USER="ec2-user"
EC2_HOST="your-ec2-instance-ip"
EC2_KEY_PATH="path/to/your-key.pem"

# Check if key file exists
if [ ! -f "$EC2_KEY_PATH" ]; then
    echo "Error: EC2 key file not found at $EC2_KEY_PATH"
    exit 1
fi

# SSH into the EC2 instance and check the status
echo "Checking the status of the Mental Health Predictor Backend on EC2..."
ssh -i "$EC2_KEY_PATH" "$EC2_USER@$EC2_HOST" << 'EOF'
echo "=== System Status ==="
uptime
echo

echo "=== Docker Status ==="
if command -v docker &> /dev/null; then
    docker ps
    echo
    docker logs --tail 20 mental-health-predictor-backend
else
    echo "Docker is not installed"
fi
echo

echo "=== Service Status ==="
if [ -f /etc/systemd/system/mental-health-predictor.service ]; then
    sudo systemctl status mental-health-predictor
else
    echo "Mental Health Predictor service is not installed"
fi
echo

echo "=== Nginx Status ==="
if command -v nginx &> /dev/null; then
    sudo systemctl status nginx
    echo
    sudo nginx -t
else
    echo "Nginx is not installed"
fi
echo

echo "=== API Test ==="
curl -s -X POST http://localhost:8000/health | jq || echo "API is not responding or jq is not installed"
EOF

echo "Status check completed!"