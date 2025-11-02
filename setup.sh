#!/bin/bash
# Quick setup script for Linux/Mac users
# This script sets up the forecaster project automatically

echo "========================================"
echo "Sales Forecaster - Quick Setup (Linux/Mac)"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed!"
    echo "Please install Python 3.8+ first"
    exit 1
fi

echo "[1/5] Python detected: $(python3 --version)"
echo ""

# Run the setup script
echo "[2/5] Running setup script..."
python3 setup.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Setup script failed!"
    exit 1
fi
echo ""

# Install dependencies
echo "[3/5] Installing dependencies..."
echo "This may take a few minutes..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[WARNING] Some dependencies may have failed to install"
    echo "Please check the error messages above"
fi
echo ""

# Check if .env exists
echo "[4/5] Checking configuration..."
if [ ! -f .env ]; then
    echo "[WARNING] .env file not found!"
    echo "Please create .env from .env.example and add your API keys"
else
    echo "[OK] .env file exists"
fi
echo ""

# Final instructions
echo "[5/5] Setup complete!"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. Edit .env file and add your GROQ_API_KEY"
echo "2. Run the server:    python3 main.py"
echo "3. Or run dashboard:  streamlit run streamlit_app.py"
echo "========================================"
echo ""
