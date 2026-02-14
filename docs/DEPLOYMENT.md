# Deployment Guide

Guide for deploying Anki Atlas in production environments.

## Quick Start

### Docker Compose (Recommended)

```yaml
# docker-compose.yml
services:
  api:
    image: anki-atlas:latest
    ports:
      - "8000:8000"
    environment:
      - ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:secret@postgres:5432/ankiatlas
      - ANKIATLAS_QDRANT_URL=http://qdrant:6333
      - ANKIATLAS_REDIS_URL=redis://redis:6379/0
      - ANKIATLAS_DEBUG=false
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_started
      redis:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=ankiatlas
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=ankiatlas
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ankiatlas"]
      interval: 5s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
  qdrant_data:
```

### Start Services

```bash
docker compose up -d
```

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `ANKIATLAS_POSTGRES_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `ANKIATLAS_QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `ANKIATLAS_REDIS_URL` | Redis URL used by arq workers | `redis://localhost:6379/0` |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | `sk-...` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `ANKIATLAS_API_HOST` | `0.0.0.0` | API bind address |
| `ANKIATLAS_API_PORT` | `8000` | API port |
| `ANKIATLAS_DEBUG` | `false` | Enable debug logging |
| `ANKIATLAS_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `ANKIATLAS_EMBEDDING_DIMENSION` | `1536` | Embedding vector dimension |
| `ANKIATLAS_QDRANT_QUANTIZATION` | `scalar` | Quantization: none, scalar, binary |
| `ANKIATLAS_QDRANT_ON_DISK` | `false` | Store vectors on disk |
| `ANKIATLAS_JOB_QUEUE_NAME` | `ankiatlas_jobs` | arq queue name |
| `ANKIATLAS_JOB_MAX_RETRIES` | `3` | Max retries for failed jobs |
| `ANKIATLAS_JOB_RESULT_TTL_SECONDS` | `86400` | Job metadata retention |

## Health Checks

### Endpoints

- `/health` - Basic health (always returns 200)
- `/ready` - Readiness check (verifies dependencies)

### Monitoring with curl

```bash
# Basic health
curl http://localhost:8000/health

# Readiness with dependencies
curl http://localhost:8000/ready
```

### Response Format

```json
{
  "status": "ready",
  "checks": {
    "postgres": "ok",
    "qdrant": "ok"
  }
}
```

## Initial Setup

### 1. Run Migrations

```bash
docker compose exec api anki-atlas migrate
```

### 2. Sync Anki Collection

```bash
# Copy collection to container or mount volume
docker compose exec api anki-atlas sync --source /data/collection.anki2
```

### 3. Verify Index

```bash
curl http://localhost:8000/index/info
```

## Production Configuration

### Security

1. **Use strong database password:**
   ```yaml
   environment:
     - POSTGRES_PASSWORD=${DB_PASSWORD}  # From secrets manager
   ```

2. **Restrict network access:**
   ```yaml
   services:
     postgres:
       networks:
         - internal
       # No ports exposed externally
   ```

3. **Enable TLS for API:**
   Use a reverse proxy (nginx, traefik) with TLS termination.

### Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          memory: 512M

  postgres:
    deploy:
      resources:
        limits:
          memory: 1G

  qdrant:
    deploy:
      resources:
        limits:
          memory: 4G  # Adjust based on collection size
```

### Memory Optimization for Qdrant

For large collections (>100k notes):

```yaml
environment:
  - ANKIATLAS_QDRANT_QUANTIZATION=scalar  # 75% memory reduction
  - ANKIATLAS_QDRANT_ON_DISK=true  # Use disk storage
```

### Logging

```yaml
services:
  api:
    environment:
      - ANKIATLAS_DEBUG=false  # JSON output for structured logging
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## Backup and Restore

### PostgreSQL

```bash
# Backup
docker compose exec postgres pg_dump -U ankiatlas ankiatlas > backup.sql

# Restore
cat backup.sql | docker compose exec -T postgres psql -U ankiatlas ankiatlas
```

### Qdrant

```bash
# Backup (snapshot API)
curl -X POST http://localhost:6333/collections/anki_notes/snapshots

# List snapshots
curl http://localhost:6333/collections/anki_notes/snapshots

# Restore from snapshot
curl -X PUT http://localhost:6333/collections/anki_notes/snapshots/recover \
  -H "Content-Type: application/json" \
  -d '{"location": "file:///qdrant/snapshots/<snapshot-name>"}'
```

## Scaling Considerations

### Single Node

- PostgreSQL: Use connection pooling (pgbouncer)
- Qdrant: Enable quantization, tune memory
- API: Run multiple uvicorn workers

```bash
uvicorn apps.api.main:app --workers 4
```

### Multi-Node (Future)

- PostgreSQL: Use managed service (RDS, Cloud SQL)
- Qdrant: Use Qdrant Cloud or distributed mode
- API: Load balance across multiple instances

## Monitoring

### Prometheus Metrics (Planned)

```yaml
services:
  api:
    ports:
      - "9090:9090"  # Metrics endpoint
```

### Example Grafana Dashboard

1. Request rate and latency
2. Error rate by endpoint
3. Database connection pool status
4. Qdrant collection size and query latency

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues.

### Common Production Issues

1. **Connection pool exhaustion:** Increase pool size or add connection limits
2. **Slow queries:** Check Qdrant index status, ensure filters use indexed fields
3. **Memory pressure:** Enable quantization, increase container memory
  worker:
    image: anki-atlas:latest
    command: ["arq", "apps.worker.WorkerSettings"]
    environment:
      - ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:secret@postgres:5432/ankiatlas
      - ANKIATLAS_QDRANT_URL=http://qdrant:6333
      - ANKIATLAS_REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_started
      redis:
        condition: service_started
