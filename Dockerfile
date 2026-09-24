# CPU serving image for the interior tagger.
#   docker build -t cvdl .                 # serve target (default)
#   docker build --target test -t cvdl-test .
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# opencv-python (a grad-cam dependency) needs libGL and glib at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY . .
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /models \
 && chown app:app /models

ENV MODEL_PATH=/models/phase2_ema.pth \
    STRICT_STARTUP=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    THREADS=4

FROM base AS test
COPY requirements-test.txt .
RUN pip install -r requirements-test.txt
USER app
CMD ["python", "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"]

FROM base AS serve
USER app
VOLUME ["/models"]
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=4)" || exit 1
CMD ["python", "serve.py"]
