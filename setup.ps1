# Mental Health Detection System Setup Script for Windows
Write-Host "Setting up Mental Health Detection System..." -ForegroundColor Green

# Create directory structure
Write-Host "Creating directory structure..." -ForegroundColor Yellow
$directories = @(
    "data\raw",
    "data\processed", 
    "data\labeled",
    "models",
    "logs",
    "reports",
    "notebooks",
    "src\data_collection",
    "src\preprocessing",
    "src\features",
    "src\models",
    "src\evaluation",
    "src\ethics",
    "src\visualization",
    "src\labeling",
    "config"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Gray
    }
}

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "requirements.txt not found. Creating it..." -ForegroundColor Red
    # Create requirements.txt if it doesn't exist
    @"
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
nltk>=3.7
textstat>=0.7.0
tweepy>=4.12.0
praw>=7.6.0
dash>=2.7.0
plotly>=5.11.0
seaborn>=0.11.0
matplotlib>=3.5.0
jupyter>=1.0.0
python-dotenv>=0.19.0
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8
    
    pip install -r requirements.txt
}

# Download NLTK data
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
python -c @"
import nltk
print('Downloading NLTK data...')
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
nltk.download('vader_lexicon')
print('NLTK data download complete!')
"@

# Create sample config file
Write-Host "Creating configuration files..." -ForegroundColor Yellow
if (!(Test-Path "config\api_config.json")) {
    @"
{
    "twitter": {
        "bearer_token": "YOUR_BEARER_TOKEN_HERE",
        "api_key": "YOUR_API_KEY_HERE",
        "api_secret": "YOUR_API_SECRET_HERE",
        "access_token": "YOUR_ACCESS_TOKEN_HERE",
        "access_token_secret": "YOUR_ACCESS_TOKEN_SECRET_HERE"
    },
    "reddit": {
        "client_id": "YOUR_CLIENT_ID_HERE",
        "client_secret": "YOUR_CLIENT_SECRET_HERE",
        "user_agent": "MentalHealthResearch/1.0"
    }
}
"@ | Out-File -FilePath "config\api_config.json" -Encoding UTF8
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Update config\api_config.json with your API credentials" -ForegroundColor White
Write-Host "2. Run: python run_pipeline.py" -ForegroundColor White
Write-Host ""
Write-Host "To get API credentials:" -ForegroundColor Cyan
Write-Host "- Twitter: https://developer.twitter.com/" -ForegroundColor White
Write-Host "- Reddit: https://www.reddit.com/prefs/apps" -ForegroundColor White
# Run this command in PowerShell
.\setup.ps1