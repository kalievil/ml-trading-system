#!/bin/bash

# ML Trading API Server Startup Script

echo "🚀 Starting ML Trading API Server..."
echo "📊 Loading trained models from user_data/models/"
echo "🌐 API will be available at http://localhost:8000"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if models directory exists
if [ ! -d "user_data/models" ]; then
    echo "❌ Models directory not found: user_data/models/"
    echo "Please make sure your trained models are in the correct location."
    exit 1
fi

# Install requirements if needed
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found"
    exit 1
fi

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "🤖 Starting FastAPI server..."
echo "📈 ML Trading Engine with XGBoost models"
echo "🔗 Webapp should connect to: http://localhost:8000"
echo ""

# Start the API server
python3 trading_api.py
