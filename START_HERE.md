# 🎯 NEW USER? START HERE!

If you just downloaded or received this project as a ZIP file, follow these simple steps:

## Quick Start (Windows)

1. **Double-click** `setup.bat`
2. Wait for installation to complete
3. Edit `.env` file (add your API key)
4. Run: `python main.py`

## Quick Start (Linux/Mac)

```bash
# Make the script executable
chmod +x setup.sh

# Run it
./setup.sh

# Edit .env file (add your API key)
nano .env

# Run the server
python3 main.py
```

## Manual Setup

If the automatic setup doesn't work:

```bash
# 1. Run the Python setup script
python setup.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and edit environment file
copy .env.example .env   # Windows
# or
cp .env.example .env     # Linux/Mac

# 4. Edit .env and add your GROQ_API_KEY

# 5. Run the application
python main.py
```

## 📖 Full Documentation

- **Setup Guide**: See `SETUP_GUIDE.md` for detailed instructions
- **Main README**: See `README.md` for full project documentation
- **API Docs**: http://localhost:8000/docs (after starting the server)

## ⚠️ Requirements

- Python 3.8 or higher
- Windows, Linux, or macOS
- Internet connection (for installing dependencies)
- Groq API key (get one at https://console.groq.com/)

## 🐛 Problems?

See `SETUP_GUIDE.md` for troubleshooting tips!
