# Deploying the Mental Health Predictor Frontend

This guide will help you deploy the frontend of the Mental Health Predictor application to GitHub Pages.

## Prerequisites

1. **Node.js and npm**: Make sure you have Node.js and npm installed on your machine. You can download them from [nodejs.org](https://nodejs.org/).

2. **GitHub Account**: You need a GitHub account to deploy to GitHub Pages.

3. **Git**: Make sure you have Git installed on your machine. You can download it from [git-scm.com](https://git-scm.com/).

## Deployment Steps

### 1. Create a GitHub Repository

1. Go to [github.com](https://github.com) and sign in to your account.
2. Click on the "+" icon in the top right corner and select "New repository".
3. Name your repository "mental-health-predictor".
4. Choose whether to make it public or private.
5. Click "Create repository".

### 2. Update the Homepage URL

1. Open the `frontend/package.json` file.
2. Update the `homepage` field with your GitHub username:
   ```json
   "homepage": "https://yourusername.github.io/mental-health-predictor"
   ```
   Replace `yourusername` with your actual GitHub username.

### 3. Initialize Git and Push to GitHub

1. Open a terminal or command prompt.
2. Navigate to the project directory:
   ```
   cd path/to/mental-health-predictor
   ```
3. Initialize a Git repository:
   ```
   git init
   ```
4. Add all files to the repository:
   ```
   git add .
   ```
5. Commit the changes:
   ```
   git commit -m "Initial commit"
   ```
6. Add the remote repository:
   ```
   git remote add origin https://github.com/yourusername/mental-health-predictor.git
   ```
   Replace `yourusername` with your actual GitHub username.
7. Push to GitHub:
   ```
   git push -u origin main
   ```

### 4. Deploy to GitHub Pages

#### Option 1: Using the Deployment Script

1. Run the deployment script:
   ```
   deploy-frontend.bat
   ```
   This script will install dependencies, build the frontend, and deploy it to GitHub Pages.

#### Option 2: Manual Deployment

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```
2. Install dependencies:
   ```
   npm install
   ```
3. Build the frontend:
   ```
   npm run build
   ```
4. Deploy to GitHub Pages:
   ```
   npm run deploy
   ```

### 5. Verify the Deployment

1. Go to your GitHub repository.
2. Click on the "Settings" tab.
3. Scroll down to the "GitHub Pages" section.
4. You should see a message saying "Your site is published at https://yourusername.github.io/mental-health-predictor".
5. Click on the link to view your deployed frontend.

## Troubleshooting

### Error: "gh-pages is not recognized as an internal or external command"

This error occurs when the `gh-pages` package is not installed. Run the following command to install it:
```
npm install --save-dev gh-pages
```

### Error: "Failed to get remote.origin.url"

This error occurs when the Git repository is not properly configured. Make sure you have added the remote repository:
```
git remote add origin https://github.com/yourusername/mental-health-predictor.git
```

### Error: "The process cannot access the file because it is being used by another process"

This error occurs when a file is locked by another process. Close any applications that might be using the file and try again.

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [React Deployment Documentation](https://create-react-app.dev/docs/deployment/#github-pages)
- [gh-pages npm package](https://www.npmjs.com/package/gh-pages)