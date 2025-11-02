# 🚀 Quick Setup Guide

## For Users Who Downloaded/Received a ZIP File

If you received this project as a ZIP file, follow these steps to get it running:

### 1️⃣ Extract and Navigate
```bash
# Extract the ZIP file
# Navigate to the project folder
cd forecaster
```

### 2️⃣ Run Setup Script (Recommended)
```bash
# This will create all required directories and check dependencies
python setup.py
```

The setup script will:
- ✅ Create all required cache directories
- ✅ Create `.env` file from example
- ✅ Verify directory structure
- ✅ Check installed dependencies

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Important:** If you get errors with `prophet` or `xgboost` on Windows:
```bash
# Try installing these separately first:
pip install prophet
pip install xgboost

# Then install the rest:
pip install -r requirements.txt
```

### 4️⃣ Configure Environment
Edit the `.env` file and add your API key:
```bash
GROQ_API_KEY=your_actual_api_key_here
```

### 5️⃣ Run the Application

**Option A: FastAPI Server**
```bash
python main.py
```
Then open: http://localhost:8000/docs

**Option B: Streamlit Dashboard**
```bash
streamlit run streamlit_app.py
```

---

## 🔧 Manual Setup (If Script Fails)

If the setup script doesn't work, create directories manually:

### Windows (Command Prompt):
```cmd
mkdir cache\uploads cache\forecasts cache\monthly cache\yearly cache\summary cache\memories cache\test
mkdir tmp_uploads
mkdir logs
copy .env.example .env
```

### Linux/Mac:
```bash
mkdir -p cache/{uploads,forecasts,monthly,yearly,summary,memories,test}
mkdir -p tmp_uploads logs
cp .env.example .env
```

---

## ❓ Troubleshooting

### Issue: "No module named 'prophet'"
**Solution:**
```bash
pip install prophet
# Or on Windows with conda:
conda install -c conda-forge prophet
```

### Issue: "No module named 'xgboost'"
**Solution:**
```bash
pip install xgboost
```

### Issue: Upload fails with "directory not found"
**Solution:**
```bash
# Run the setup script:
python setup.py

# Or manually create directories (see Manual Setup above)
```

### Issue: "Session not found" after upload
**Solution:**
Check that these files exist:
- `cache/sessions.json` (should be created automatically)
- `tmp_uploads/` directory exists

### Issue: Import errors with statsmodels/ARIMA
**Solution:**
```bash
pip install statsmodels --upgrade
```

---

## 📦 What's Included in a Fresh Download

Required directories (created by setup script):
- `cache/` - Forecast caching system
  - `uploads/` - Upload tracking
  - `forecasts/` - Cached forecasts
  - `monthly/` - Monthly frequency cache
  - `yearly/` - Yearly frequency cache
  - `summary/` - Summary data cache
- `tmp_uploads/` - Temporary file storage during uploads
- `logs/` - Application logs

Required files:
- `.env` - Environment configuration (copy from `.env.example`)
- `requirements.txt` - Python dependencies

---

## 🎯 Quick Test

After setup, test if everything works:

```bash
# Test 1: Run setup verification
python setup.py

# Test 2: Check imports
python -c "import pandas, numpy, fastapi, streamlit; print('✅ Core imports OK')"

# Test 3: Check optional imports
python -c "import prophet, xgboost; print('✅ ML libraries OK')"

# Test 4: Start the server
python main.py
```

---

## 📝 Common First-Time Setup Checklist

- [ ] Extracted ZIP file
- [ ] Ran `python setup.py`
- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] Created/updated `.env` file with GROQ_API_KEY
- [ ] Verified directories exist (cache/, tmp_uploads/, logs/)
- [ ] Tested server starts: `python main.py`
- [ ] Can access API docs: http://localhost:8000/docs

---

## 🆘 Still Having Issues?

1. **Check Python version**: Requires Python 3.8+
   ```bash
   python --version
   ```

2. **Use a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Clear any old cache**:
   ```bash
   # Delete and recreate cache directories
   python setup.py
   ```

4. **Check logs**:
   - Look in `logs/` directory for error messages
   - Check console output when running the app

---

## 🎉 You're Ready!

Once setup is complete, you can:
- Upload CSV/Excel files via the API
- Generate forecasts for sales and quantities
- Use the SQL chatbot for data analysis
- View forecasts in the Streamlit dashboard

Enjoy! 🚀
