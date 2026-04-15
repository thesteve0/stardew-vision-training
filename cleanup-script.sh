#!/bin/bash

PROJECT_NAME="stardew-vision-training"

echo "Cleaning up $PROJECT_NAME ROCm development environment..."

# Detect container runtime (Docker or Podman)
if command -v docker &> /dev/null; then
    RUNTIME="docker"
elif command -v podman &> /dev/null; then
    RUNTIME="podman"
else
    echo "Error: Neither docker nor podman found"
    exit 1
fi

echo "Using container runtime: $RUNTIME"

# Stop any running devcontainer
echo "Stopping any running devcontainers..."
$RUNTIME ps -q --filter "label=devcontainer.metadata" | xargs -r $RUNTIME stop

echo ""
echo "Cleanup complete."
echo ""
echo "Your data directories are preserved:"
echo "  - models/"
echo "  - datasets/"
echo "  - .cache/"
echo ""
echo "To delete data: rm -rf models/* datasets/* .cache/*"