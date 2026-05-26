FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements early for layer caching
COPY requirements.txt .

# Install CPU-only PyTorch (much lighter than CUDA build; HF free tier is CPU-only)
RUN pip install --no-cache-dir \
    "torch==2.1.0+cpu" "torchvision==0.16.0+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (FastAPI, OpenCV, Pillow, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Pre-download DINOv2 ViT-S/14 weights at build time (~86 MB) to avoid
# cold-start delay on first request in HuggingFace Spaces.
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')" || true

# HuggingFace Spaces requires port 7860
EXPOSE 7860

ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PORT=7860

CMD ["python", "-m", "uvicorn", "app.web.server:app", "--host", "0.0.0.0", "--port", "7860"]
