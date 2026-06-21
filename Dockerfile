# Use a lightweight python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set up work directory and set ownership
WORKDIR /app
RUN chown user:user /app

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy requirements file first for caching
COPY --chown=user:user requirements-docker.txt /app/requirements-docker.txt

# Install PyTorch CPU-only package (saves ~1.5GB of image size)
RUN pip3 install --no-cache-dir --user torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other python requirements
RUN pip3 install --no-cache-dir --user -r /app/requirements-docker.txt

# Pre-download NLTK resources and ASR model weights to avoid runtime downloads/timeouts
RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
RUN python3 -c "from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC; Wav2Vec2Processor.from_pretrained('MihirRPatil/nptel-asr-phoneme-v3'); Wav2Vec2ForCTC.from_pretrained('MihirRPatil/nptel-asr-phoneme-v3')"

# Copy the entire project codebase
COPY --chown=user:user . /app

# Generate the Prisma client for Python
RUN prisma generate --schema=/app/product/backend/prisma/schema.prisma

# Expose the default Hugging Face Spaces port
EXPOSE 7860

# Command to run the application
CMD ["python3", "product/backend/app.py"]
