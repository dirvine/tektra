"""Database configuration and session management."""

from typing import AsyncGenerator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import settings

# Convert sync postgres URL to async
database_url = settings.database_url
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create async engine
engine = create_async_engine(
    database_url,
    echo=settings.debug,
    future=True,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create async session factory
async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Metadata for table creation
metadata = MetaData()

# Base class for all models
Base = declarative_base(metadata=metadata)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_database() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        # Import all models to ensure they're registered
        from .models.user import User  # noqa: F401
        from .models.conversation import Conversation, Message  # noqa: F401
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session() -> AsyncSession:
    """Get a database session (for WebSocket use)."""
    return async_session_factory()


async def close_database() -> None:
    """Close database connections."""
    await engine.dispose()