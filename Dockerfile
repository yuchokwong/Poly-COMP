# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Choose CPU or CUDA torch wheels at build time:
#   CPU (default): --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
#   CUDA 12.1    : --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    TSR_CONFIG=/app/configs/default.yaml

WORKDIR /app

# System deps: Pillow needs libjpeg/zlib (bundled), and OpenCV-style libs are not
# required, but libgl/libglib are handy if users extend the pipeline.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install torch first (from the requested index) so requirements.txt is satisfied.
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "tsr.api:app", "--host", "0.0.0.0", "--port", "8000"]
