FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
RUN python -m pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest", "-q"]
