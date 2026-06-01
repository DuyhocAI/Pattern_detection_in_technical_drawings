# Docker Deployment Guide

## Quick Start

### 1. Build Image

```bash
docker build -t bom-detector:latest .
```

Or use the build script:
```bash
chmod +x build-docker.sh
./build-docker.sh
```

### 2. Run Container

**Local development (port 8000):**
```bash
docker run -p 8000:7860 bom-detector:latest
```

Open http://localhost:8000

**Detached mode:**
```bash
docker run -d -p 8000:7860 --name bom-detector bom-detector:latest
```

**With volume mounts (development):**
```bash
docker run -p 8000:7860 \
  -v $(pwd)/app:/app/app \
  -v $(pwd)/src:/app/src \
  bom-detector:latest
```

### 3. Using Docker Compose

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down

# View logs
docker-compose logs -f bom-detector
```

---

## Production Deployment

### HuggingFace Spaces

The Dockerfile is pre-configured for HuggingFace Spaces:

```yaml
---
title: BOM Pattern Detection
sdk: docker
app_port: 7860
---
```

**Deploy steps:**
1. Fork/clone to HuggingFace Spaces
2. Set `sdk: docker` in README.md frontmatter
3. Push code — HuggingFace will auto-build and deploy

### Docker Hub

**Build and push:**
```bash
docker build -t your-username/bom-detector:latest .
docker push your-username/bom-detector:latest
```

**Users can then run:**
```bash
docker pull your-username/bom-detector:latest
docker run -p 8000:7860 your-username/bom-detector:latest
```

### AWS ECR

```bash
# Create repository
aws ecr create-repository --repository-name bom-detector

# Get login token
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag bom-detector:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/bom-detector:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/bom-detector:latest
```

---

## Image Details

### Base Image
- **OS:** Python 3.11-slim (Debian-based, minimal)
- **Size:** ~1.5 GB (includes PyTorch CPU + DINOv2)

### Pre-installed Weights
- **DINOv2 ViT-S/14** — Downloaded at build time (~86 MB)
- **Qwen2-VL-2B** — Lazy-loaded on first use with `use_vlm=True` (~5 GB)

### Environment Variables
```bash
KMP_DUPLICATE_LIB_OK=TRUE  # OpenMP compatibility
PORT=7860                  # FastAPI port (HuggingFace Spaces standard)
```

### Exposed Ports
- **7860** — FastAPI Web UI (HuggingFace standard)

---

## Health Check

```bash
# Test container is responding
curl http://localhost:8000/api/health

# Expected response
{"status": "ok"}
```

---

## Troubleshooting

### Docker daemon not running
```bash
# Windows
# Start Docker Desktop from Start Menu

# Linux
sudo systemctl start docker

# macOS
open -a Docker
```

### Container exits immediately
```bash
# View logs
docker logs <container-id>

# Run with interactive mode to see errors
docker run -it bom-detector:latest
```

### Out of disk space (large models)
```bash
# Clean up Docker images/containers
docker system prune -a

# Or be more selective
docker image prune
docker container prune
```

### Port already in use
```bash
# Use different port
docker run -p 9000:7860 bom-detector:latest

# Then open http://localhost:9000
```

---

## Development Workflow

### 1. Build with cache for fast iteration
```bash
docker build -t bom-detector:dev .
```

### 2. Mount source for live editing
```bash
docker run -it -p 8000:7860 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/app:/app/app \
  bom-detector:dev
```

### 3. Rebuild when needed
```bash
docker-compose up --build
```

---

## Performance Tips

1. **Use Docker Desktop resource limits** (Settings → Resources)
   - **Memory:** 4-8 GB
   - **CPUs:** 4+
   - **Disk:** 20+ GB (for models cache)

2. **Enable BuildKit for faster builds**
   ```bash
   export DOCKER_BUILDKIT=1
   docker build -t bom-detector:latest .
   ```

3. **Layer caching**
   - requirements.txt copied early for cache hits
   - DINOv2 weights pre-downloaded at build time

---

## Cleanup

```bash
# Stop container
docker stop bom-detector

# Remove container
docker rm bom-detector

# Remove image
docker rmi bom-detector:latest

# Remove all dangling images
docker image prune -f

# Full cleanup (⚠️ removes all images/containers)
docker system prune -a
```

---

## Next Steps

- **Deploy to HuggingFace Spaces** — Free GPU tier available
- **Push to Docker Hub** — Public distribution
- **Custom registry** — AWS ECR, Google Artifact Registry, etc.

For more info, see [README.md](README.md) and [System Specification](design_spec/system_spec.html).
