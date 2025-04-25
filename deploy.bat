@echo off
echo ===== Mental Health Predictor Frontend Deployment =====
echo.

cd frontend

echo Installing dependencies...
call npm install
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install dependencies.
    echo Make sure Node.js is properly installed and you're running this script in a new terminal.
    pause
    exit /b 1
)

echo.
echo Building and deploying to GitHub Pages...
call npm run deploy
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to deploy to GitHub Pages.
    echo Check the error messages above for more information.
    pause
    exit /b 1
)

echo.
echo ===== Deployment Complete =====
echo.
echo Your frontend has been deployed to GitHub Pages.
echo It may take a few minutes for the changes to be visible.
echo.
echo Please visit: https://chaithanya.github.io/mental-health-predictor
echo.
pause