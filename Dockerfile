FROM rust:1.85-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY bins ./bins

RUN cargo build --release --bin anki-atlas-api

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /app/target/release/anki-atlas-api /usr/local/bin/

USER appuser

EXPOSE 8000

CMD ["anki-atlas-api"]
