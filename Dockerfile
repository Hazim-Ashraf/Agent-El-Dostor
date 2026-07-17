# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Built with BuildKit (default in Docker 23+ / Compose v2). The `--mount=type=cache`
# blocks below persist the apt + pip DOWNLOAD caches BETWEEN builds, so changing a
# dependency re-downloads ONLY the new package instead of the whole stack. Do NOT set
# PIP_NO_CACHE_DIR here — it would defeat the pip cache mount.
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    PYTHONPATH=/app

# System deps:
#  - Tesseract (Arabic+English OCR) + poppler (PDF->image for OCR)  [ingestion]
#  - Pango/Cairo/gdk-pixbuf: WeasyPrint's rendering backend (RTL Arabic shaping via
#    HarfBuzz) for contract-generation PDF export
#  - fonts-hosny-amiri (the Amiri Arabic naskh font — family name "Amiri" — for legal
#    Arabic; package is named `fonts-hosny-amiri` on Debian trixie) + fonts-dejavu-core (Latin)
# The apt package lists + .deb archives live in cache mounts (not baked into the image),
# so a later change re-fetches only new packages. `rm docker-clean` keeps apt from
# auto-deleting the .debs we want to cache.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ara \
        tesseract-ocr-eng \
        poppler-utils \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        fonts-hosny-amiri \
        fonts-dejavu-core

WORKDIR /app

# Install CPU-only PyTorch FIRST. This project never uses a GPU (Docker on macOS has no
# MPS/CUDA passthrough anyway), and on x86_64 the default PyPI `torch` pulls ~2.5 GB of
# NVIDIA CUDA libraries. Forcing the CPU wheel keeps the image ~2.5 GB smaller. On arm64
# (M1) PyPI's torch is already CPU-only, so we install it normally there.
# TARGETARCH is provided automatically by BuildKit (amd64 / arm64).
ARG TARGETARCH
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch ; \
    else \
        pip install torch ; \
    fi

# The rest of the Python deps in their own layer, keyed ONLY on pyproject.toml (so code
# edits don't reinstall). torch is already satisfied above, so it isn't re-pulled. The
# pip cache mount means a dependency change downloads ONLY new wheels.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; open('/tmp/requirements.txt','w').write(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" && \
    pip install -r /tmp/requirements.txt

COPY . .

EXPOSE 8501

# Default: run the Streamlit testing GUI. Ingestion is run via `docker compose run app ...`.
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
