from prisma import Prisma

db = Prisma()

async def get_db() -> Prisma:
    """Dependency provider that yields the connected Prisma instance."""
    if not db.is_connected():
        await db.connect()
    return db
