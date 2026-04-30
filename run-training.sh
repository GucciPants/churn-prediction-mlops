#!/bin/bash
# Docker-based training script for Linux/Mac
# Usage: ./run-training.sh

echo "Starting MLflow and training services..."
docker-compose --profile training up --build

echo ""
echo "Training completed! Models saved to ./models/"
echo "MLflow UI available at http://localhost:5000"
