# Build context: repository root (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/backend

COPY backend/requirements.lock backend/pyproject.toml ./
RUN pip install -r requirements.lock

# Neural embedding / reranking models are optional (CPU-only baseline runs without them):
#   docker compose build --build-arg INSTALL_ML=1 backend
ARG INSTALL_ML=0
COPY backend/requirements-ml.lock ./
RUN if [ "$INSTALL_ML" = "1" ]; then pip install -r requirements-ml.lock; fi

COPY backend/ ./
COPY configs/ /srv/configs/
RUN pip install --no-deps -e . && useradd --create-home app && mkdir -p /srv/artifacts /srv/data && chown -R app /srv
USER app

EXPOSE 8000
# Schema migrations run before the API starts.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
