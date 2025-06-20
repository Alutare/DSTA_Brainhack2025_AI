# Dockerfile for building the OCR image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM python:3.12-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Copy tessdata
COPY src/tessdata /usr/share/tesseract-ocr/4.00/tessdata/

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn ocr_server:app --port 5003 --host 0.0.0.0
