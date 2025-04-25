# Frontend Deployment Steps

Follow these steps to deploy the Mental Health Predictor frontend to GitHub Pages:

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in to your account
2. Click on the "+" icon in the top right corner and select "New repository"
3. Name your repository "mental-health-predictor"
4. Choose whether to make it public or private
5. Click "Create repository"

## Step 2: Initialize Git and Push to GitHub

Open a new Command Prompt or PowerShell window and run the following commands:

```bash
# Navigate to the project directory
cd C:\Users\chaithanya\Downloads\mental-health-predictor

# Initialize Git repository
git init

# Add all files to Git
git add .

# Commit the changes
git commit -m "Initial commit"

# Add the remote repository (replace 'chaithanya' with your GitHub username if different)
git remote add origin https://github.com/chaithanya/mental-health-predictor.git

# Push to GitHub
git push -u origin main
```

If you encounter an error with the last command, try:

```bash
git push -u origin master
```

## Step 3: Install Dependencies and Deploy

After successfully pushing to GitHub, run the following commands:

```bash
# Navigate to the frontend directory
cd C:\Users\chaithanya\Downloads\mental-health-predictor\frontend

# Install dependencies
npm install

# Deploy to GitHub Pages
npm run deploy
```

This will build the frontend and deploy it to the gh-pages branch of your repository.

## Step 4: Configure GitHub Pages

1. Go to your GitHub repository
2. Click on "Settings"
3. Scroll down to the "GitHub Pages" section
4. Make sure the source is set to the "gh-pages" branch
5. Click "Save"

## Step 5: Access Your Deployed Frontend

After a few minutes, your frontend will be available at:
https://chaithanya.github.io/mental-health-predictor

## Troubleshooting

If you encounter any issues:

1. Make sure Node.js and npm are properly installed
2. Check that you have the correct permissions to push to the GitHub repository
3. Ensure that the gh-pages package is installed as a dev dependency
4. Verify that the homepage URL in package.json matches your GitHub username

For more detailed instructions, refer to the DEPLOYMENT.md file.