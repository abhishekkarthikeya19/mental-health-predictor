@echo off
echo ===== Mental Health Predictor Git Initialization =====
echo.
echo This script will help you initialize a Git repository and push it to GitHub.
echo.

set /p username=Enter your GitHub username: 

echo.
echo Step 1: Initializing Git repository...
git init
if %ERRORLEVEL% neq 0 (
    echo Error initializing Git repository.
    exit /b 1
)
echo Git repository initialized successfully.
echo.

echo Step 2: Updating package.json with your GitHub username...
powershell -Command "(Get-Content frontend/package.json) -replace 'yourusername', '%username%' | Set-Content frontend/package.json"
if %ERRORLEVEL% neq 0 (
    echo Error updating package.json.
    exit /b 1
)
echo package.json updated successfully.
echo.

echo Step 3: Adding files to Git...
git add .
if %ERRORLEVEL% neq 0 (
    echo Error adding files to Git.
    exit /b 1
)
echo Files added successfully.
echo.

echo Step 4: Committing changes...
git commit -m "Initial commit"
if %ERRORLEVEL% neq 0 (
    echo Error committing changes.
    exit /b 1
)
echo Changes committed successfully.
echo.

echo Step 5: Adding remote repository...
git remote add origin https://github.com/%username%/mental-health-predictor.git
if %ERRORLEVEL% neq 0 (
    echo Error adding remote repository.
    exit /b 1
)
echo Remote repository added successfully.
echo.

echo Step 6: Pushing to GitHub...
git push -u origin main
if %ERRORLEVEL% neq 0 (
    echo Error pushing to GitHub. You might need to authenticate or create the repository first.
    echo Please create a repository named "mental-health-predictor" on GitHub and try again.
    exit /b 1
)
echo.
echo ===== Git Initialization Complete =====
echo.
echo Your code has been pushed to GitHub.
echo.
echo Next steps:
echo 1. Run deploy-frontend.bat to deploy the frontend to GitHub Pages.
echo 2. Visit https://%username%.github.io/mental-health-predictor to see your deployed frontend.
echo.
pause