FROM rust:1.86-slim AS builder
RUN apt-get update && apt-get install -y python3-dev python3-pip pkg-config libssl-dev clang libclang-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages maturin
WORKDIR /app
COPY . .
RUN maturin build --release -m chromapro_core/Cargo.toml

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/chromapro_core/target/wheels/*.whl /tmp/
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl
ENV CHROMAPRO_PATH=/data
ENV PERSIST_DIRECTORY=/data
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "from chromapro import PersistentClient; c=PersistentClient('/data'); print(c.heartbeat()); c.close()"
CMD ["python", "-c", "import time; print('chromapro image ready'); time.sleep(3600)"]
