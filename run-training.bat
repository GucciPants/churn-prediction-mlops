@echo off
REM Docker-based training script for Windows
REM Usage: run-training.bat

echo Starting MLflow and training services...
docker-compose --profile training up --build

echo.
echo Training completed! Models saved to ./models/
echo MLflow UI available at http://localhost:5000
pause
