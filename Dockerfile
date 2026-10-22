# Slim image: Debian-based (wheels for psycopg/argon2 just work, unlike Alpine's musl).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies before copying all the code, so Docker reuses this layer
# when only source files change.
# CPU-only torch first: the default Linux wheel bundles ~2 GB of CUDA libraries
# this image never uses (embeddings + reranking run on CPU).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config.yaml ./
COPY prompts ./prompts

# Don't run as root: if the app is ever compromised, the attacker isn't root in the container.
RUN useradd --create-home --uid 1000 scout \
    && mkdir -p /home/scout/.cache/huggingface && chown -R scout /home/scout/.cache
USER scout

EXPOSE 8000
CMD ["uvicorn", "scout.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
