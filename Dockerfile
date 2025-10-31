FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps (minimal) and pip requirements
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

# Ensure Python output is unbuffered (helpful for logs)
ENV PYTHONUNBUFFERED=1

# Expose ports used by FastAPI and Streamlit
EXPOSE 8000 8501

# Default command: run FastAPI (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
