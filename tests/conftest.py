import os
import sys

# Ensure the repository root (project) is in sys.path so pytest can import local packages
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optional: expose a pytest fixture for quick access to project root
import pytest


@pytest.fixture(scope="session")
def project_root():
    return ROOT
