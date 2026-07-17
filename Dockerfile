FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    PYTHONPATH=/app

# System deps: Tesseract (Arabic+English OCR) + poppler (PDF->image for OCR).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ara \
        tesseract-ocr-eng \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies in their own cached layer (parsed from pyproject.toml)
# so code edits don't trigger a full torch reinstall.
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    python -c "import tomllib; open('/tmp/requirements.txt','w').write(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" && \
    pip install -r /tmp/requirements.txt

COPY . .

EXPOSE 8501

# Default: run the Streamlit testing GUI. Ingestion is run via `docker compose run app ...`.
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
