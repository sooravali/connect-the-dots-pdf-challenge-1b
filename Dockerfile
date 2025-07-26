# Multi-stage Docker build for persona-driven document analysis
# Challenge 1B - Adobe Hackathon 2025

# Build stage - optimized for fast builds and proper model caching
FROM --platform=linux/amd64 python:3.9-slim as builder

WORKDIR /build

# Set environment variables for build optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SENTENCE_TRANSFORMERS_HOME=/build/model_cache \
    TRANSFORMERS_CACHE=/build/model_cache/transformers \
    HF_HOME=/build/model_cache/huggingface \
    XDG_CACHE_HOME=/build/model_cache

# Install system dependencies - grouped into one RUN to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /build/model_cache

# Copy requirements first for better caching
COPY requirements.txt .

# Install packages with CPU-only torch and other dependencies
# Combined into a single RUN command for smaller image layers
RUN pip install --no-cache-dir --user torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir --user -r requirements.txt

# Pre-download and cache the sentence transformer model in a dedicated directory
# Create a simple script to ensure proper caching
RUN echo "from sentence_transformers import SentenceTransformer; \
          import os; \
          print('Cache directories:', \
          os.environ.get('SENTENCE_TRANSFORMERS_HOME'), \
          os.environ.get('TRANSFORMERS_CACHE')); \
          model = SentenceTransformer('all-MiniLM-L6-v2'); \
          print('Model loaded successfully and cached to:', \
          os.environ.get('SENTENCE_TRANSFORMERS_HOME'))" > /build/cache_model.py \
    && python /build/cache_model.py \
    && ls -la /build/model_cache

# Runtime stage - optimized for small size and performance
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Set environment variables for runtime optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    SENTENCE_TRANSFORMERS_HOME=/root/model_cache \
    TRANSFORMERS_CACHE=/root/model_cache/transformers \
    HF_HOME=/root/model_cache/huggingface \
    XDG_CACHE_HOME=/root/model_cache \
    PATH=/root/.local/bin:$PATH

# Copy Python packages and pre-cached models from builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/model_cache /root/model_cache

# Copy only the required application code
COPY process_persona.py pdf_section_extractor.py semantic_analyzer.py ./

# Create input/output directories
RUN mkdir -p /app/input /app/output

# Run the processing script
CMD ["python", "process_persona.py"]
