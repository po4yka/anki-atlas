"""PostgreSQL database connection and utilities."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from packages.common.config import Settings, get_settings
from packages.common.logging import get_logger

logger = get_logger(module=__name__)

# Module-level pool (initialized on first use)
_pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None


async def get_pool(
    settings: Settings | None = None,
) -> AsyncConnectionPool[AsyncConnection[dict[str, Any]]]:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        if settings is None:
            settings = get_settings()
        _pool = AsyncConnectionPool(
            conninfo=settings.postgres_url,
            min_size=2,
            max_size=10,
            open=False,
            kwargs={"row_factory": dict_row},
        )
        await _pool.open()
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_connection(
    settings: Settings | None = None,
) -> AsyncGenerator[AsyncConnection[dict[str, Any]]]:
    """Get a database connection from the pool."""
    pool = await get_pool(settings)
    async with pool.connection() as conn:
        yield conn


@dataclass
class MigrationResult:
    """Result of a migration run."""

    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


async def run_migrations(settings: Settings | None = None) -> MigrationResult:
    """Run all pending migrations.

    Tracks applied migrations in a `schema_migrations` table so each
    .sql file executes at most once.  Returns a MigrationResult with
    the names of applied and skipped migrations.
    """
    if settings is None:
        settings = get_settings()

    migrations_dir = Path(__file__).parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))

    result = MigrationResult()

    async with await psycopg.AsyncConnection.connect(
        settings.postgres_url,
        row_factory=dict_row,
    ) as conn:
        # Bootstrap the tracking table
        async with conn.cursor() as cur:
            await cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "  name TEXT PRIMARY KEY,"
                "  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                ")"
            )
        await conn.commit()

        # Load already-applied set
        async with conn.cursor() as cur:
            await cur.execute("SELECT name FROM schema_migrations")
            rows = await cur.fetchall()
        already_applied: set[str] = {row["name"] for row in rows}

        for migration_file in migration_files:
            migration_name = migration_file.stem

            if migration_name in already_applied:
                logger.debug("migration_skipped", name=migration_name)
                result.skipped.append(migration_name)
                continue

            sql = migration_file.read_text()

            async with conn.cursor() as cur:
                await cur.execute(sql)
                await cur.execute(
                    "INSERT INTO schema_migrations (name) VALUES (%s)",
                    (migration_name,),
                )
            await conn.commit()
            logger.info("migration_applied", name=migration_name)
            result.applied.append(migration_name)

    return result


async def check_connection(settings: Settings | None = None) -> bool:
    """Check if database is reachable."""
    if settings is None:
        settings = get_settings()

    try:
        async with (
            await psycopg.AsyncConnection.connect(
                settings.postgres_url,
                connect_timeout=5,
            ) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT 1")
        return True
    except psycopg.OperationalError as exc:
        logger.warning("postgres_health_check_failed", error=str(exc))
        return False
