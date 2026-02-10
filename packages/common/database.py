"""PostgreSQL database connection and utilities."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from packages.common.config import Settings, get_settings

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


async def run_migrations(settings: Settings | None = None) -> list[str]:
    """Run all pending migrations.

    Returns list of applied migration names.
    """
    if settings is None:
        settings = get_settings()

    migrations_dir = Path(__file__).parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))

    applied: list[str] = []

    async with await psycopg.AsyncConnection.connect(
        settings.postgres_url,
        row_factory=dict_row,
    ) as conn:
        for migration_file in migration_files:
            migration_name = migration_file.stem
            sql = migration_file.read_text()

            # Execute migration
            async with conn.cursor() as cur:
                await cur.execute(sql)

            await conn.commit()
            applied.append(migration_name)

    return applied


async def check_connection(settings: Settings | None = None) -> bool:
    """Check if database is reachable."""
    if settings is None:
        settings = get_settings()

    try:
        async with await psycopg.AsyncConnection.connect(
            settings.postgres_url,
            connect_timeout=5,
        ) as conn, conn.cursor() as cur:
            await cur.execute("SELECT 1")
        return True
    except Exception:
        return False
