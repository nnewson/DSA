FROM rust:1.85-slim

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/

RUN cargo build --tests

ENTRYPOINT ["cargo", "test"]
