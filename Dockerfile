FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libglib2.0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (separate layer for caching)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining deps
COPY requirements.txt .
RUN grep -v "^torch\|^torchvision" requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt

# Copy project source
COPY . .

# Pre-download DINOv2 ViT-S/14 weights at build time (~86 MB)
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')" || true

# HuggingFace Spaces requires port 7860
EXPOSE 7860

ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PORT=7860

CMD ["python", "-m", "uvicorn", "app.web.server:app", "--host", "0.0.0.0", "--port", "7860"]
