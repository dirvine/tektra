"""
Conversation Management Service

Handles:
- Conversation persistence and retrieval
- Message history and context management
- User conversation organization
- Context-aware AI interactions
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.conversation import Conversation, Message, MessageRole, MessageType
from ..models.user import User

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversations and message history."""

    def __init__(self):
        self.max_context_messages = 50  # Maximum messages to include in AI context
        self.context_window_tokens = 4000  # Rough token limit for context

    async def create_conversation(
        self,
        db: AsyncSession,
        user_id: int,
        title: Optional[str] = None,
        model_name: str = "phi-3-mini",
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            user_id=user_id,
            title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            model_name=model_name,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

        logger.info(f"Created conversation {conversation.id} for user {user_id}")
        return conversation

    async def get_conversation(
        self, db: AsyncSession, conversation_id: int, user_id: Optional[int] = None
    ) -> Optional[Conversation]:
        """Get a conversation by ID."""
        query = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )

        if user_id:
            query = query.where(Conversation.user_id == user_id)

        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_user_conversations(
        self, db: AsyncSession, user_id: int, limit: int = 50, offset: int = 0
    ) -> List[Conversation]:
        """Get all conversations for a user."""
        query = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
            .offset(offset)
        )

        result = await db.execute(query)
        return result.scalars().all()

    async def add_message(
        self,
        db: AsyncSession,
        conversation_id: int,
        role: MessageRole,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to a conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_type=message_type,
            message_metadata=metadata or {},
            created_at=datetime.utcnow(),
        )

        db.add(message)

        # Update conversation's updated_at timestamp
        conversation_query = select(Conversation).where(
            Conversation.id == conversation_id
        )
        conversation_result = await db.execute(conversation_query)
        conversation = conversation_result.scalar_one_or_none()

        if conversation:
            conversation.updated_at = datetime.utcnow()

            # Auto-generate title from first user message if not set
            if not conversation.title or conversation.title.startswith("Conversation"):
                if role == MessageRole.USER and len(content.strip()) > 0:
                    # Use first 50 characters as title
                    title = content.strip()[:50]
                    if len(content) > 50:
                        title += "..."
                    conversation.title = title

        await db.commit()
        await db.refresh(message)

        logger.info(f"Added {role.value} message to conversation {conversation_id}")
        return message

    async def get_conversation_messages(
        self,
        db: AsyncSession,
        conversation_id: int,
        limit: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Message]:
        """Get messages from a conversation."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )

        if not include_system:
            query = query.where(Message.role != MessageRole.SYSTEM)

        if limit:
            query = query.limit(limit)

        result = await db.execute(query)
        return result.scalars().all()

    async def get_context_for_ai(
        self, db: AsyncSession, conversation_id: int, max_messages: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get conversation context formatted for AI model."""
        max_messages = max_messages or self.max_context_messages

        # Get recent messages for context
        messages = await self.get_conversation_messages(
            db, conversation_id, limit=max_messages, include_system=False
        )

        # Format for AI service
        context = []
        for message in messages:
            context.append(
                {
                    "role": message.role.value,
                    "content": message.content,
                    "timestamp": (
                        message.created_at.isoformat() if message.created_at else None
                    ),
                }
            )

        return context

    async def delete_conversation(
        self, db: AsyncSession, conversation_id: int, user_id: Optional[int] = None
    ) -> bool:
        """Delete a conversation and all its messages."""
        query = select(Conversation).where(Conversation.id == conversation_id)

        if user_id:
            query = query.where(Conversation.user_id == user_id)

        result = await db.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            return False

        await db.delete(conversation)
        await db.commit()

        logger.info(f"Deleted conversation {conversation_id}")
        return True

    async def update_conversation_title(
        self,
        db: AsyncSession,
        conversation_id: int,
        title: str,
        user_id: Optional[int] = None,
    ) -> bool:
        """Update conversation title."""
        query = select(Conversation).where(Conversation.id == conversation_id)

        if user_id:
            query = query.where(Conversation.user_id == user_id)

        result = await db.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            return False

        conversation.title = title
        conversation.updated_at = datetime.utcnow()

        await db.commit()
        return True

    async def search_conversations(
        self, db: AsyncSession, user_id: int, query: str, limit: int = 20
    ) -> List[Conversation]:
        """Search conversations by title or content."""
        # Simple text search - can be enhanced with full-text search later
        search_query = (
            select(Conversation)
            .where(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.title.ilike(f"%{query}%"),
                )
            )
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
        )

        result = await db.execute(search_query)
        return result.scalars().all()

    async def get_conversation_stats(
        self, db: AsyncSession, conversation_id: int
    ) -> Dict[str, Any]:
        """Get statistics about a conversation."""
        messages = await self.get_conversation_messages(db, conversation_id)

        stats = {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if m.role == MessageRole.USER]),
            "assistant_messages": len(
                [m for m in messages if m.role == MessageRole.ASSISTANT]
            ),
            "total_characters": sum(len(m.content) for m in messages),
            "first_message": messages[0].created_at.isoformat() if messages else None,
            "last_message": messages[-1].created_at.isoformat() if messages else None,
        }

        return stats

    async def cleanup_old_conversations(
        self, db: AsyncSession, user_id: int, days_old: int = 90
    ) -> int:
        """Clean up old conversations (for maintenance)."""
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        query = select(Conversation).where(
            and_(Conversation.user_id == user_id, Conversation.updated_at < cutoff_date)
        )

        result = await db.execute(query)
        old_conversations = result.scalars().all()

        count = 0
        for conversation in old_conversations:
            await db.delete(conversation)
            count += 1

        await db.commit()

        logger.info(f"Cleaned up {count} old conversations for user {user_id}")
        return count


# Global conversation manager instance
conversation_manager = ConversationManager()
