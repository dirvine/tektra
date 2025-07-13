#!/usr/bin/env docker
# Tektra AI Assistant - Production Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG TEKTRA_VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies with UV
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md LICENCE ./

# Install Tektra in production mode
RUN uv pip install -e .

# Production stage
FROM python:3.11-slim as production

# Set metadata labels
LABEL org.label-schema.name="Tektra AI Assistant" \
      org.label-schema.description="Production-ready AI assistant with SmolAgents integration" \
      org.label-schema.version=$TEKTRA_VERSION \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0"

# Create non-root user for security
RUN groupadd -r tektra && useradd -r -g tektra -d /app -s /sbin/nologin tektra

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    jq \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/README.md ./

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config /app/temp \
    && chown -R tektra:tektra /app

# Copy configuration templates
COPY docker/config/ ./config/
COPY docker/entrypoint.sh ./entrypoint.sh
COPY docker/healthcheck.sh ./healthcheck.sh

# Make scripts executable
RUN chmod +x ./entrypoint.sh ./healthcheck.sh

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TEKTRA_ENV=production \
    TEKTRA_DATA_DIR=/app/data \
    TEKTRA_LOGS_DIR=/app/logs \
    TEKTRA_CONFIG_DIR=/app/config \
    TEKTRA_TEMP_DIR=/app/temp

# Expose ports
EXPOSE 8000 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ./healthcheck.sh

# Switch to non-root user
USER tektra

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["serve"]