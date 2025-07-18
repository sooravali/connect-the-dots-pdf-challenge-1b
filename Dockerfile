# Multi-stage Docker build for persona-driven document analysis
# Challenge 1B - Adobe Hackathon 2025

# Build stage
FROM --platform=linux/amd64 python:3.9-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .

# Install packages with CPU-only torch
RUN pip install --no-cache-dir --user torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir --user -r requirements.txt

# Download and cache the sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Runtime stage
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY process_persona.py .
COPY pdf_section_extractor.py .
COPY semantic_analyzer.py .

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false

# Run the processing script
CMD ["python", "process_persona.py"]
