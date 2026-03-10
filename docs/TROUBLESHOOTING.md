# Troubleshooting Guide

Common issues and their solutions when running Anki Atlas.

## Connection Issues

### "Cannot connect to PostgreSQL"

**Symptoms:**
- `DatabaseConnectionError` on startup
- `/ready` endpoint shows postgres: failed

**Solutions:**

1. **Check PostgreSQL is running:**
   ```bash
   # Docker
   docker ps | grep postgres

   # Systemd
   sudo systemctl status postgresql
   ```

2. **Verify connection URL:**
   ```bash
   echo $ANKIATLAS_POSTGRES_URL
   # Should be: postgresql://user:pass@host:5432/database
   ```

3. **Test connection manually:**
   ```bash
   psql $ANKIATLAS_POSTGRES_URL -c "SELECT 1"
   ```

4. **Check network connectivity:**
   ```bash
   nc -zv localhost 5432
   ```

### "Cannot connect to Qdrant"

**Symptoms:**
- `VectorStoreConnectionError` during indexing
- `/ready` endpoint shows qdrant: failed

**Solutions:**

1. **Check Qdrant is running:**
   ```bash
   docker ps | grep qdrant
   ```

2. **Verify URL:**
   ```bash
   echo $ANKIATLAS_QDRANT_URL
   # Should be: http://localhost:6333
   ```

3. **Check health endpoint:**
   ```bash
   curl http://localhost:6333/healthz
   ```

4. **Check container logs:**
   ```bash
   docker logs qdrant 2>&1 | tail -20
   ```

### "Port already in use"

**Symptoms:**
- `OSError: [Errno 98] Address already in use` on API startup

**Solutions:**

1. **Find what's using the port:**
   ```bash
   lsof -i :8000
   # or
   ss -tlnp | grep 8000
   ```

2. **Kill the process or use different port:**
   ```bash
   # Kill by PID
   kill <pid>

   # Or use different port
   ANKIATLAS_API_PORT=8001 cargo run --bin anki-atlas-api
   ```

## Embedding Issues

### "Embedding API timeout"

**Symptoms:**
- Indexing hangs or fails with timeout
- `EmbeddingTimeoutError` in logs

**Solutions:**

1. **Check API rate limits:**
   - OpenAI has rate limits; reduce batch size
   - Wait and retry after rate limit window

2. **Check network connectivity:**
   ```bash
   curl -v https://api.openai.com/v1/embeddings
   ```

3. **Verify API key:**
   ```bash
   echo $OPENAI_API_KEY | head -c 10
   ```

4. **Retry with a smaller operational scope:**
   Reduce the data set you are reprocessing or enqueue indexing after a narrower sync.

### "Collection dimension mismatch"

**Symptoms:**
- Error: `Collection has dimension X, but provider requires Y`
- Happens when changing embedding models

**Solutions:**

1. **Enqueue a fresh sync followed by index work:**
   ```bash
   curl -X POST http://localhost:8000/jobs/sync \
     -H "Content-Type: application/json" \
     -d '{"source":"/path/to/collection.anki2","force_reindex":true}'
   ```

2. **Manually delete collection:**
   ```bash
   curl -X DELETE http://localhost:6333/collections/anki_notes
   ```

## Sync Issues

### "Collection file not found"

**Symptoms:**
- `CollectionNotFoundError` when syncing

**Solutions:**

1. **Verify path exists:**
   ```bash
   ls -la /path/to/collection.anki2
   ```

2. **Find Anki collection location:**
   ```bash
   # macOS
   ls ~/Library/Application\ Support/Anki2/*/collection.anki2

   # Linux
   ls ~/.local/share/Anki2/*/collection.anki2

   # Windows (PowerShell)
   ls "$env:APPDATA\Anki2\*\collection.anki2"
   ```

3. **Ensure Anki is closed:**
   Anki locks the database while running. Close Anki before syncing.

### "Database migration failed"

**Symptoms:**
- `MigrationError` on startup or sync
- Schema version mismatch

**Solutions:**

1. **Run migrations manually:**
   ```bash
   anki-atlas migrate
   ```

2. **Check migration status:**
   ```bash
   # Connect to database and check
   psql $ANKIATLAS_POSTGRES_URL -c "SELECT * FROM migrations"
   ```

3. **Reset database (last resort):**
   ```bash
   # WARNING: This will delete all data
   psql $ANKIATLAS_POSTGRES_URL -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
   anki-atlas migrate
   ```

## Indexing and Retrieval Issues

### "Indexed data looks incomplete"

**Symptoms:**
- Search or downstream analytics return empty results
- Expected cards not appearing

**Solutions:**

1. **Verify the service is ready:**
   ```bash
   curl http://localhost:8000/ready
   ```

2. **Check collection data exists in PostgreSQL:**
   ```bash
   psql $ANKIATLAS_POSTGRES_URL -c "SELECT COUNT(*) FROM notes"
   ```

3. **Check vectors exist in Qdrant:**
   ```bash
   curl http://localhost:6333/collections/anki_notes | jq .result.points_count
   ```

4. **If counts are zero, enqueue a fresh sync job:**
   ```bash
   curl -X POST http://localhost:8000/jobs/sync \
     -H "Content-Type: application/json" \
     -d '{"source":"/path/to/collection.anki2"}'
   ```

### "Retrieval is slow"

**Symptoms:**
- Search takes more than 2 seconds
- Timeout errors

**Solutions:**

1. **Check Qdrant status:**
   ```bash
   curl http://localhost:6333/collections/anki_notes | jq .status
   # Should be "green"
   ```

2. **Check system resources:**
   ```bash
   htop
   # Look for high CPU/memory usage
   ```

## Performance Issues

### "High memory usage"

**Symptoms:**
- Qdrant consuming excessive RAM
- OOM errors

**Solutions:**

1. **Enable quantization (already default):**
   Check `ANKIATLAS_QDRANT_QUANTIZATION=scalar` is set.

2. **Enable on-disk storage:**
   ```bash
   export ANKIATLAS_QDRANT_ON_DISK=true
   curl -X POST http://localhost:8000/jobs/index \
     -H "Content-Type: application/json" \
     -d '{"force_reindex":true}'
   ```

3. **Increase Qdrant memory limit:**
   ```yaml
   # docker-compose.yml
   services:
     qdrant:
       deploy:
         resources:
           limits:
             memory: 2G
   ```

## Getting Help

If these solutions don't resolve your issue:

1. Check the logs with debug mode:
   ```bash
   ANKIATLAS_DEBUG=true anki-atlas <command>
   ```

2. Open an issue with:
   - Full error message
   - Steps to reproduce
   - Environment details
   - Relevant logs
