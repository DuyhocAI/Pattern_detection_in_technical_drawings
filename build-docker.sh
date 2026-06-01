#!/bin/bash

# BOM Pattern Detection — Docker Build Script
# Build, test, and optionally push to Docker Hub

set -e

IMAGE_NAME="bom-detector"
IMAGE_TAG="latest"
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_USER="${DOCKER_USER:-your-username}"

echo "=========================================="
echo "🐳 BOM Pattern Detection — Docker Builder"
echo "=========================================="
echo ""

# Step 1: Build image
echo "[1/4] Building Docker image..."
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""

docker build -t "$IMAGE_NAME:$IMAGE_TAG" .

if [ $? -eq 0 ]; then
    echo "✓ Image built successfully"
else
    echo "✗ Build failed"
    exit 1
fi

# Step 2: Show image info
echo ""
echo "[2/4] Image info:"
docker images "$IMAGE_NAME:$IMAGE_TAG"

# Step 3: Test run
echo ""
echo "[3/4] Testing container..."
echo "Starting container on port 8000..."
echo "You can test at: http://localhost:8000"
echo ""
echo "To stop: Press Ctrl+C"
echo ""

docker run --rm -p 8000:7860 "$IMAGE_NAME:$IMAGE_TAG" &
PID=$!

sleep 5

echo "Testing API health endpoint..."
if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✓ Container health check passed"
    kill $PID 2>/dev/null || true
else
    echo "⚠ Health check failed (may still work, network timeout)"
    kill $PID 2>/dev/null || true
fi

# Step 4: Optional push to registry
echo ""
echo "[4/4] Docker Hub push (optional)"
echo ""
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Tagging for Docker Hub..."
    docker tag "$IMAGE_NAME:$IMAGE_TAG" "$REGISTRY/$DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG"

    echo "Pushing to $REGISTRY/$DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG..."
    docker push "$REGISTRY/$DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG"

    echo "✓ Pushed successfully"
    echo ""
    echo "Now available as:"
    echo "  docker pull $REGISTRY/$DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG"
else
    echo "Skipped push to registry"
fi

echo ""
echo "=========================================="
echo "✓ Docker build complete!"
echo "=========================================="
echo ""
echo "To run locally:"
echo "  docker run -p 8000:7860 bom-detector:latest"
echo ""
echo "Then open:"
echo "  http://localhost:8000"
echo ""
