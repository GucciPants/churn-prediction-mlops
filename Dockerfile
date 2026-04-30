# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

# Install bash and git for potential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
