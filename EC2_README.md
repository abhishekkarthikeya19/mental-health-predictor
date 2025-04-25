# Mental Health Predictor Backend - EC2 Deployment

This README provides instructions for deploying the Mental Health Predictor backend to an Amazon EC2 instance.

## Prerequisites

1. **AWS Account**: You need an AWS account to create and manage EC2 instances.
2. **EC2 Instance**: Launch an EC2 instance with Amazon Linux 2 or Ubuntu.
3. **Security Group**: Configure the security group to allow inbound traffic on ports 22 (SSH), 80 (HTTP), and 8000 (API).
4. **Key Pair**: Create or use an existing key pair for SSH access to your EC2 instance.

## Deployment Options

### Option 1: Using the Windows Batch Script

1. Open the `deploy_to_ec2.bat` file in the project root directory.
2. Run the script by double-clicking it or executing it from the command prompt.
3. Follow the prompts to enter your EC2 instance details.

### Option 2: Using the Bash Script (Linux/Mac)

1. Open the `deploy_to_ec2.sh` script in the project root directory.
2. Update the following variables with your own values:
   - `EC2_USER`: The username for your EC2 instance (usually `ec2-user` for Amazon Linux or `ubuntu` for Ubuntu).
   - `EC2_HOST`: The public IP address or DNS of your EC2 instance.
   - `EC2_KEY_PATH`: The path to your key pair file (.pem).
3. Make the script executable:
   ```bash
   chmod +x deploy_to_ec2.sh
   ```
4. Run the script:
   ```bash
   ./deploy_to_ec2.sh
   ```

### Option 3: Manual Deployment (Without Docker)

If you prefer not to use Docker, you can use the manual deployment script:

1. Open the `deploy_to_ec2_manual.sh` script in the project root directory.
2. Update the variables as described in Option 2.
3. Make the script executable:
   ```bash
   chmod +x deploy_to_ec2_manual.sh
   ```
4. Run the script:
   ```bash
   ./deploy_to_ec2_manual.sh
   ```

## Checking Deployment Status

To check the status of your deployment:

1. Open the `check_ec2_status.sh` script in the project root directory.
2. Update the variables as described in Option 2.
3. Make the script executable:
   ```bash
   chmod +x check_ec2_status.sh
   ```
4. Run the script:
   ```bash
   ./check_ec2_status.sh
   ```

## Detailed Deployment Guide

For a more detailed guide on deploying to EC2, please refer to the `EC2_DEPLOYMENT_GUIDE.md` file in the project root directory.

## Troubleshooting

If you encounter any issues during deployment, please check the following:

1. Make sure your EC2 instance is running and accessible.
2. Verify that the security group allows inbound traffic on the required ports.
3. Check that your key pair file has the correct permissions (chmod 400 on Linux/Mac).
4. Review the logs on the EC2 instance for any error messages.

## Support

If you need further assistance, please open an issue on the GitHub repository or contact the project maintainers.