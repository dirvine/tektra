#!/usr/bin/env python3
"""Test database initialization."""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database():
    """Test database connection and table creation."""
    try:
        from app.database import init_database, get_db
        from app.models import User, Conversation, Message
        
        logger.info("Initializing database...")
        await init_database()
        logger.info("✅ Database initialized successfully")
        
        # Test database session
        logger.info("Testing database session...")
        async for db in get_db():
            logger.info("✅ Database session created successfully")
            break
        
        logger.info("✅ All database tests passed")
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_database())