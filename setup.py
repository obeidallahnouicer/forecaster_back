"""
Setup script to initialize the forecaster project.

Run this script after downloading/cloning the project to ensure
all required directories and configuration files are in place.
"""

import os
from pathlib import Path
import sys


def create_directory_structure():
    """Create all required directories."""
    project_root = Path(__file__).parent
    
    directories = [
        # Cache directories
        "cache/uploads",
        "cache/forecasts",
        "cache/monthly",
        "cache/yearly",
        "cache/summary",
        "cache/memories",
        "cache/test",
        # Temporary uploads
        "tmp_uploads",
        # Logs
        "logs",
        # Server data (if needed)
        "server_data",
    ]
    
    print("🔧 Creating directory structure...")
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    print("✅ Directory structure created successfully!\n")


def check_env_file():
    """Check if .env file exists, create from example if not."""
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    print("🔧 Checking environment configuration...")
    
    if env_file.exists():
        print("  ✓ .env file already exists")
    elif env_example.exists():
        print("  ⚠ .env file not found, copying from .env.example")
        env_file.write_text(env_example.read_text())
        print("  ✓ .env file created - PLEASE UPDATE WITH YOUR API KEYS!")
    else:
        print("  ⚠ .env.example not found - creating basic .env")
        env_file.write_text("""# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration (optional)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sales_db
DB_USER=postgres
DB_PASSWORD=your_password_here

# Application Settings
LOG_LEVEL=INFO
CACHE_TTL=86400
""")
        print("  ✓ Basic .env file created - PLEASE UPDATE WITH YOUR API KEYS!")
    
    print()


def check_requirements():
    """Check if requirements are installed."""
    project_root = Path(__file__).parent
    requirements_file = project_root / "requirements.txt"
    
    print("🔧 Checking Python dependencies...")
    
    if not requirements_file.exists():
        print("  ❌ requirements.txt not found!")
        return False
    
    try:
        import pandas
        import numpy
        import fastapi
        import streamlit
        print("  ✓ Core dependencies appear to be installed")
        
        # Check optional dependencies
        try:
            import prophet
            print("  ✓ Prophet is installed")
        except ImportError:
            print("  ⚠ Prophet not found - install with: pip install prophet")
        
        try:
            import xgboost
            print("  ✓ XGBoost is installed")
        except ImportError:
            print("  ⚠ XGBoost not found - install with: pip install xgboost")
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            print("  ✓ Statsmodels is installed")
        except ImportError:
            print("  ⚠ Statsmodels not found - should be included in requirements")
            
    except ImportError as e:
        print(f"  ❌ Missing core dependencies: {e}")
        print(f"\n  Run: pip install -r requirements.txt")
        return False
    
    print()
    return True


def create_test_cache_files():
    """Create initial cache files if they don't exist."""
    project_root = Path(__file__).parent
    
    print("🔧 Initializing cache files...")
    
    # Create sessions.json if it doesn't exist
    sessions_file = project_root / "cache" / "sessions.json"
    if not sessions_file.exists():
        import json
        sessions_file.write_text(json.dumps({}, indent=2))
        print("  ✓ cache/sessions.json created")
    
    print()


def verify_setup():
    """Verify the setup is complete."""
    project_root = Path(__file__).parent
    
    print("🔍 Verifying setup...")
    
    checks = {
        "cache directory": (project_root / "cache").exists(),
        "tmp_uploads directory": (project_root / "tmp_uploads").exists(),
        "logs directory": (project_root / "logs").exists(),
        ".env file": (project_root / ".env").exists(),
        "requirements.txt": (project_root / "requirements.txt").exists(),
        "main.py": (project_root / "main.py").exists(),
    }
    
    all_good = True
    for check_name, result in checks.items():
        status = "✓" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_good = False
    
    print()
    return all_good


def main():
    """Main setup routine."""
    print("=" * 60)
    print("🚀 Sales Forecaster - Project Setup")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Create directories
        create_directory_structure()
        
        # Step 2: Setup environment file
        check_env_file()
        
        # Step 3: Check dependencies
        deps_ok = check_requirements()
        
        # Step 4: Create cache files
        create_test_cache_files()
        
        # Step 5: Verify setup
        setup_ok = verify_setup()
        
        # Final status
        print("=" * 60)
        if setup_ok and deps_ok:
            print("✅ Setup completed successfully!")
            print()
            print("Next steps:")
            print("  1. Update .env file with your API keys")
            print("  2. Run the API server: python main.py")
            print("  3. Or run the dashboard: streamlit run streamlit_app.py")
        else:
            print("⚠️  Setup completed with warnings")
            print()
            print("Please address the issues above before running the application.")
            if not deps_ok:
                print("\nInstall dependencies:")
                print("  pip install -r requirements.txt")
        print("=" * 60)
        
        return 0 if (setup_ok and deps_ok) else 1
        
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
