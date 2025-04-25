# Deploying the Mental Health Predictor Backend to Amazon EC2

This guide will walk you through the process of deploying the Mental Health Predictor backend to an Amazon EC2 instance.

## Prerequisites

1. **AWS Account**: You need an AWS account to create and manage EC2 instances.
2. **EC2 Instance**: Launch an EC2 instance with Amazon Linux 2 or Ubuntu.
3. **Security Group**: Configure the security group to allow inbound traffic on ports 22 (SSH), 80 (HTTP), and 8000 (API).
4. **Key Pair**: Create or use an existing key pair for SSH access to your EC2 instance.

## Step 1: Launch an EC2 Instance

1. Sign in to the AWS Management Console and open the Amazon EC2 console.
2. Click "Launch Instance".
3. Choose an Amazon Machine Image (AMI) - Amazon Linux 2 is recommended.
4. Choose an instance type - t2.micro is eligible for the free tier and should be sufficient for testing.
5. Configure instance details as needed.
6. Add storage - the default 8GB should be sufficient.
7. Add tags if desired.
8. Configure the security group to allow inbound traffic on ports 22 (SSH), 80 (HTTP), and 8000 (API).
9. Review and launch the instance.
10. Select an existing key pair or create a new one, and launch the instance.

## Step 2: Connect to Your EC2 Instance

### For Windows Users:

1. Open PuTTY or another SSH client.
2. Enter your instance's public DNS or IP address.
3. Configure the SSH connection to use your key pair.
4. Connect to the instance.

### For Mac/Linux Users:

```bash
ssh -i /path/to/your-key.pem ec2-user@your-ec2-instance-ip
```

## Step 3: Prepare the Deployment Script

1. Open the `deploy_to_ec2.sh` script in the project root directory.
2. Update the following variables with your own values:
   - `EC2_USER`: The username for your EC2 instance (usually `ec2-user` for Amazon Linux or `ubuntu` for Ubuntu).
   - `EC2_HOST`: The public IP address or DNS of your EC2 instance.
   - `EC2_KEY_PATH`: The path to your key pair file (.pem).

```bash
# Configuration - Replace these values with your own
EC2_USER="ec2-user"
EC2_HOST="your-ec2-instance-ip"
EC2_KEY_PATH="path/to/your-key.pem"
```

## Step 4: Run the Deployment Script

1. Open a terminal or command prompt on your local machine.
2. Navigate to the project root directory.
3. Make the deployment script executable:

```bash
chmod +x deploy_to_ec2.sh
```

4. Run the deployment script:

```bash
./deploy_to_ec2.sh
```

The script will:
- Copy the necessary files to your EC2 instance
- Install Docker and Docker Compose
- Build and start the Docker container
- Set up Nginx as a reverse proxy (if needed)

## Step 5: Verify the Deployment

1. Wait for the deployment to complete.
2. Test the API by sending a request to your EC2 instance:

```bash
curl -X POST http://your-ec2-instance-ip:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text_input": "I feel sad today"}'
```

You should receive a JSON response with the prediction result.

## Step 6: Update the Frontend Configuration

1. Update the API configuration in the frontend to point to your EC2 instance:

```javascript
// API Configuration
const API_CONFIG = {
  baseUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:8000' 
    : 'http://your-ec2-instance-ip:8000',
  endpoints: {
    predict: '/predict/',
    analyze: '/analyze/'
  }
};
```

## Step 7: Set Up HTTPS (Recommended for Production)

For production use, it's recommended to set up HTTPS using Let's Encrypt:

1. Install Certbot:

```bash
sudo amazon-linux-extras install epel -y
sudo yum install certbot python-certbot-nginx -y
```

2. Obtain and install a certificate:

```bash
sudo certbot --nginx -d your-domain.com
```

3. Follow the prompts to complete the setup.

## Troubleshooting

### Container Fails to Start

If the container fails to start, check the Docker logs:

```bash
docker logs mental-health-predictor-backend
```

### API Not Accessible

If the API is not accessible, check the following:

1. Make sure the container is running:

```bash
docker ps
```

2. Check the security group settings to ensure that port 8000 is open.

3. Check the Nginx configuration if you're using it as a reverse proxy:

```bash
sudo nginx -t
```

### Model Not Found

If you encounter a "Model not found" error, you may need to train the model:

```bash
docker exec -it mental-health-predictor-backend python /app/app/train_model.py
```

## Maintenance

### Updating the Deployment

To update the deployment with new code:

1. Update the `deploy_to_ec2.sh` script if needed.
2. Run the deployment script again:

```bash
./deploy_to_ec2.sh
```

### Monitoring the Application

You can monitor the application logs using Docker:

```bash
docker logs -f mental-health-predictor-backend
```

### Stopping the Application

To stop the application:

```bash
docker-compose down
```

### Restarting the Application

To restart the application:

```bash
docker-compose up -d
```

## Security Considerations

1. **Restrict Access**: Limit access to your EC2 instance by configuring the security group to allow only necessary inbound traffic.
2. **Use HTTPS**: Set up HTTPS using Let's Encrypt to encrypt data in transit.
3. **Keep Software Updated**: Regularly update the EC2 instance, Docker, and other software components.
4. **Monitor Logs**: Regularly check the application logs for any suspicious activity.
5. **Implement Rate Limiting**: The application already has rate limiting configured, but you may want to adjust the settings based on your needs.

## Conclusion

You have successfully deployed the Mental Health Predictor backend to an Amazon EC2 instance. The API is now accessible at `http://your-ec2-instance-ip:8000` and can be used by the frontend application.