FROM python:3.11-slim

WORKDIR /app

# libglib2.0-0 is required by opencv-python-headless on slim images.
# Without this line, import cv2 crashes instantly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render injects $PORT. No --reload in production.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
