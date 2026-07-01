import logging
from prisma import Prisma
from fastapi import HTTPException

logger = logging.getLogger(__name__)

db = Prisma()

# Tracks whether the DB was reachable at startup or on last attempt.
# Prevents hammering a dead host on every single request.
_db_available: bool = True


async def try_connect_db() -> bool:
    """Attempt to connect to the database. Returns True on success."""
    global _db_available
    if db.is_connected():
        _db_available = True
        return True
    try:
        await db.connect()
        _db_available = True
        logger.info("Prisma Database successfully connected.")
        return True
    except Exception as e:
        _db_available = False
        logger.error(
            "DATABASE CONNECTION ERROR: Failed to connect to database. "
            "Database-dependent features (auth, scoring history) will be unavailable. "
            f"Error details: {e}"
        )
        return False


async def get_db() -> Prisma:
    """FastAPI dependency that returns a connected Prisma client.

    Raises HTTP 503 if the database is unreachable, giving the frontend
    a clean error instead of an unhandled 500 crash.
    """
    if db.is_connected():
        return db

    if not _db_available:
        # Don't retry a known-dead connection on every request
        raise HTTPException(
            status_code=503,
            detail="Database is currently unavailable. Please try again later.",
        )

    # DB was available last time but got disconnected — try once
    connected = await try_connect_db()
    if not connected:
        raise HTTPException(
            status_code=503,
            detail="Database is currently unavailable. Please try again later.",
        )
    return db
