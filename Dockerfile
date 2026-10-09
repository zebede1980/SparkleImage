# SparkleImage — AI photo repair & enhancement
# Optimised for Linux ARM64 (Ampere) servers.

FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app/ ./app/
RUN pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel

# --- Runtime ---
FROM python:3.11-slim-bookworm AS runtime

LABEL description="AI photo repair, recovery and enhancement"

# opencv-python-headless needs no GUI libraries — only libgomp for its
# threading. The previous image pulled in libgl1/libsm6/libxext6 to satisfy the
# non-headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The wheel is the only copy of the application code — the previous image
# installed the wheel *and* copied app/ over the top of it.
COPY --from=builder /build/dist/*.whl ./
RUN pip install --no-cache-dir *.whl && rm *.whl

# YuNet face detector (~230KB). Without it the app falls back to OpenCV's
# bundled Haar cascade, so a failed download degrades quality rather than
# breaking detection — which is what happened to the previous face check,
# whose weights were never fetched at all.
RUN mkdir -p /app/data/models && \
    curl -fsSL -o /app/data/models/face_detection_yunet_2023mar.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    || echo "YuNet download failed — falling back to the bundled cascade"

# Run as a normal user. Bind-mounted host directories must be writable by this
# uid; docker-compose sets it from the host so the container need not run as root.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "${APP_GID}" sparkle 2>/dev/null || true && \
    useradd -u "${APP_UID}" -g "${APP_GID}" -d /app -s /sbin/nologin sparkle 2>/dev/null || true && \
    mkdir -p /app/data/uploads /app/data/output && \
    chown -R "${APP_UID}:${APP_GID}" /app
USER ${APP_UID}:${APP_GID}

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

ENTRYPOINT ["python", "-m", "app.main"]
