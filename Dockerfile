FROM rust:1.83-bookworm AS builder

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYO3_PYTHON="/opt/venv/bin/python"

RUN pip install --no-cache-dir maturin numpy

WORKDIR /app
COPY . .

RUN maturin develop --release

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY tests/ tests/
COPY benchmarks/ benchmarks/

CMD ["python", "tests/test_basic.py"]
