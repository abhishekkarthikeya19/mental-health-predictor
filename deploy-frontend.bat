@echo off
echo ===== Mental Health Predictor Frontend Deployment =====
echo.
echo This script will help you deploy the frontend to GitHub Pages.
echo.

echo Step 1: Installing dependencies...
cd frontend
call npm install
if %ERRORLEVEL% neq 0 (
    echo Error installing dependencies. Please make sure Node.js is installed.
    exit /b 1
)
echo Dependencies installed successfully.
echo.

echo Step 2: Building the frontend...
call npm run build
if %ERRORLEVEL% neq 0 (
    echo Error building the frontend.
    exit /b 1
)
echo Frontend built successfully.
echo.

echo Step 3: Deploying to GitHub Pages...
call npm run deploy
if %ERRORLEVEL% neq 0 (
    echo Error deploying to GitHub Pages.
    exit /b 1
)
echo.
echo ===== Deployment Complete =====
echo.
echo Your frontend has been deployed to GitHub Pages.
echo It may take a few minutes for the changes to be visible.
echo.
echo Please visit: https://yourusername.github.io/mental-health-predictor
echo.
echo Don't forget to update the "homepage" field in package.json with your actual GitHub username.
echo.
pause