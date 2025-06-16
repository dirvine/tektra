"""
Conversation Search and Organization Service.

Advanced search, filtering, tagging, and organization features for conversations.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from sqlalchemy import and_, or_, func, desc, asc, text
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.conversation import (
    Conversation, Message, Tag, MessageRole, MessageType, 
    ConversationCategory, conversation_tags
)
from ..models.user import User

logger = logging.getLogger(__name__)


class ConversationSearchService:
    """Service for advanced conversation search and organization."""
    
    def __init__(self):
        self.search_operators = {
            'AND': 'and_',
            'OR': 'or_',
            'NOT': 'not_',
            'CONTAINS': 'contains',
            'EXACT': 'exact',
            'STARTS': 'startswith',
            'ENDS': 'endswith'
        }
    
    async def search_conversations(
        self,
        db: AsyncSession,
        user_id: int,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[ConversationCategory] = None,
        model_name: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        is_pinned: Optional[bool] = None,
        is_archived: Optional[bool] = None,
        min_messages: Optional[int] = None,
        max_messages: Optional[int] = None,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Conversation], int]:
        """
        Advanced conversation search with multiple filters.
        
        Returns tuple of (conversations, total_count).
        """
        try:
            # Start with base query
            query_builder = db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).options(
                selectinload(Conversation.tags),
                selectinload(Conversation.messages)
            )
            
            # Apply filters
            if query:
                query_builder = self._apply_text_search(query_builder, query)
            
            if tags:
                query_builder = self._apply_tag_filter(query_builder, tags)
            
            if category:
                query_builder = query_builder.filter(Conversation.category == category)
            
            if model_name:
                query_builder = query_builder.filter(Conversation.model_name == model_name)
            
            if date_from:
                query_builder = query_builder.filter(Conversation.created_at >= date_from)
            
            if date_to:
                query_builder = query_builder.filter(Conversation.created_at <= date_to)
            
            if is_pinned is not None:
                query_builder = query_builder.filter(Conversation.is_pinned == is_pinned)
            
            if is_archived is not None:
                query_builder = query_builder.filter(Conversation.is_archived == is_archived)
            
            if min_messages is not None:
                query_builder = query_builder.filter(Conversation.message_count >= min_messages)
            
            if max_messages is not None:
                query_builder = query_builder.filter(Conversation.message_count <= max_messages)
            
            # Count total results
            total_count = await query_builder.count()
            
            # Apply sorting
            query_builder = self._apply_sorting(query_builder, sort_by, sort_order)
            
            # Apply pagination
            conversations = await query_builder.offset(offset).limit(limit).all()
            
            return conversations, total_count
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            raise
    
    async def search_messages(
        self,
        db: AsyncSession,
        user_id: int,
        query: str,
        conversation_id: Optional[int] = None,
        role: Optional[MessageRole] = None,
        message_type: Optional[MessageType] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        is_favorite: Optional[bool] = None,
        is_important: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Message], int]:
        """
        Search messages within conversations.
        
        Returns tuple of (messages, total_count).
        """
        try:
            # Base query with user filter through conversation
            query_builder = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id
            ).options(
                selectinload(Message.conversation)
            )
            
            # Apply text search
            if query:
                query_builder = query_builder.filter(
                    or_(
                        Message.content.ilike(f"%{query}%"),
                        Message.search_vector.ilike(f"%{query}%")
                    )
                )
            
            # Apply filters
            if conversation_id:
                query_builder = query_builder.filter(Message.conversation_id == conversation_id)
            
            if role:
                query_builder = query_builder.filter(Message.role == role)
            
            if message_type:
                query_builder = query_builder.filter(Message.message_type == message_type)
            
            if date_from:
                query_builder = query_builder.filter(Message.created_at >= date_from)
            
            if date_to:
                query_builder = query_builder.filter(Message.created_at <= date_to)
            
            if is_favorite is not None:
                query_builder = query_builder.filter(Message.is_favorite == is_favorite)
            
            if is_important is not None:
                query_builder = query_builder.filter(Message.is_important == is_important)
            
            # Count total results
            total_count = await query_builder.count()
            
            # Apply sorting and pagination
            messages = await query_builder.order_by(
                desc(Message.created_at)
            ).offset(offset).limit(limit).all()
            
            return messages, total_count
            
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            raise
    
    def _apply_text_search(self, query_builder, search_query: str):
        """Apply text search to conversations."""
        # Simple implementation - can be enhanced with full-text search
        search_terms = search_query.split()
        
        for term in search_terms:
            query_builder = query_builder.filter(
                or_(
                    Conversation.title.ilike(f"%{term}%"),
                    Conversation.description.ilike(f"%{term}%"),
                    Conversation.messages.any(Message.content.ilike(f"%{term}%"))
                )
            )
        
        return query_builder
    
    def _apply_tag_filter(self, query_builder, tags: List[str]):
        """Apply tag filter to conversations."""
        return query_builder.filter(
            Conversation.tags.any(Tag.name.in_(tags))
        )
    
    def _apply_sorting(self, query_builder, sort_by: str, sort_order: str):
        """Apply sorting to query."""
        sort_column = getattr(Conversation, sort_by, Conversation.updated_at)
        
        if sort_order.lower() == "desc":
            return query_builder.order_by(desc(sort_column))
        else:
            return query_builder.order_by(asc(sort_column))
    
    async def get_conversation_analytics(
        self,
        db: AsyncSession,
        user_id: int,
        conversation_id: Optional[int] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get analytics for conversations."""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            base_query = db.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.created_at >= since_date
            )
            
            if conversation_id:
                base_query = base_query.filter(Conversation.id == conversation_id)
            
            # Basic stats
            total_conversations = await base_query.count()
            
            # Message stats
            message_stats = await db.query(
                func.count(Message.id).label('total_messages'),
                func.avg(Message.tokens_used).label('avg_tokens'),
                func.avg(Message.processing_time).label('avg_processing_time')
            ).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= since_date
            ).first()
            
            # Category distribution
            category_stats = await db.query(
                Conversation.category,
                func.count(Conversation.id).label('count')
            ).filter(
                Conversation.user_id == user_id,
                Conversation.created_at >= since_date
            ).group_by(Conversation.category).all()
            
            # Model usage stats
            model_stats = await db.query(
                Conversation.model_name,
                func.count(Conversation.id).label('count')
            ).filter(
                Conversation.user_id == user_id,
                Conversation.created_at >= since_date,
                Conversation.model_name.isnot(None)
            ).group_by(Conversation.model_name).all()
            
            # Daily activity
            daily_stats = await db.query(
                func.date(Conversation.created_at).label('date'),
                func.count(Conversation.id).label('conversations'),
                func.count(Message.id).label('messages')
            ).outerjoin(Message).filter(
                Conversation.user_id == user_id,
                Conversation.created_at >= since_date
            ).group_by(func.date(Conversation.created_at)).all()
            
            return {
                "period_days": days,
                "total_conversations": total_conversations,
                "total_messages": message_stats.total_messages or 0,
                "avg_tokens_per_message": float(message_stats.avg_tokens or 0),
                "avg_processing_time_ms": float(message_stats.avg_processing_time or 0),
                "category_distribution": [
                    {"category": stat.category, "count": stat.count}
                    for stat in category_stats
                ],
                "model_usage": [
                    {"model": stat.model_name, "count": stat.count}
                    for stat in model_stats
                ],
                "daily_activity": [
                    {
                        "date": stat.date.isoformat(),
                        "conversations": stat.conversations,
                        "messages": stat.messages or 0
                    }
                    for stat in daily_stats
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation analytics: {e}")
            raise


class TagService:
    """Service for managing conversation tags."""
    
    async def create_tag(
        self,
        db: AsyncSession,
        user_id: int,
        name: str,
        color: Optional[str] = None,
        description: Optional[str] = None
    ) -> Tag:
        """Create a new tag."""
        try:
            # Check if tag already exists for this user
            existing_tag = await db.query(Tag).filter(
                Tag.user_id == user_id,
                Tag.name == name
            ).first()
            
            if existing_tag:
                raise ValueError(f"Tag '{name}' already exists")
            
            tag = Tag(
                user_id=user_id,
                name=name,
                color=color,
                description=description
            )
            
            db.add(tag)
            await db.commit()
            await db.refresh(tag)
            
            return tag
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating tag: {e}")
            raise
    
    async def get_user_tags(
        self,
        db: AsyncSession,
        user_id: int,
        include_usage: bool = True
    ) -> List[Tag]:
        """Get all tags for a user."""
        try:
            query = db.query(Tag).filter(Tag.user_id == user_id)
            
            if include_usage:
                query = query.order_by(desc(Tag.usage_count), Tag.name)
            else:
                query = query.order_by(Tag.name)
            
            return await query.all()
            
        except Exception as e:
            logger.error(f"Error getting user tags: {e}")
            raise
    
    async def add_tags_to_conversation(
        self,
        db: AsyncSession,
        conversation_id: int,
        tag_names: List[str],
        user_id: int
    ) -> bool:
        """Add tags to a conversation."""
        try:
            conversation = await db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            ).first()
            
            if not conversation:
                raise ValueError("Conversation not found")
            
            for tag_name in tag_names:
                # Get or create tag
                tag = await db.query(Tag).filter(
                    Tag.user_id == user_id,
                    Tag.name == tag_name
                ).first()
                
                if not tag:
                    tag = Tag(user_id=user_id, name=tag_name)
                    db.add(tag)
                    await db.flush()
                
                # Add tag to conversation if not already present
                if tag not in conversation.tags:
                    conversation.tags.append(tag)
                    tag.usage_count += 1
            
            await db.commit()
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error adding tags to conversation: {e}")
            raise
    
    async def remove_tags_from_conversation(
        self,
        db: AsyncSession,
        conversation_id: int,
        tag_names: List[str],
        user_id: int
    ) -> bool:
        """Remove tags from a conversation."""
        try:
            conversation = await db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            ).options(selectinload(Conversation.tags)).first()
            
            if not conversation:
                raise ValueError("Conversation not found")
            
            for tag_name in tag_names:
                tag = next((t for t in conversation.tags if t.name == tag_name), None)
                if tag:
                    conversation.tags.remove(tag)
                    tag.usage_count = max(0, tag.usage_count - 1)
            
            await db.commit()
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error removing tags from conversation: {e}")
            raise
    
    async def delete_tag(
        self,
        db: AsyncSession,
        tag_id: int,
        user_id: int
    ) -> bool:
        """Delete a tag."""
        try:
            tag = await db.query(Tag).filter(
                Tag.id == tag_id,
                Tag.user_id == user_id
            ).first()
            
            if not tag:
                raise ValueError("Tag not found")
            
            await db.delete(tag)
            await db.commit()
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting tag: {e}")
            raise


class ConversationExportService:
    """Service for exporting and importing conversations."""
    
    async def export_conversations(
        self,
        db: AsyncSession,
        user_id: int,
        conversation_ids: Optional[List[int]] = None,
        format: str = "json",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export conversations to various formats."""
        try:
            query = db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).options(
                selectinload(Conversation.messages),
                selectinload(Conversation.tags)
            )
            
            if conversation_ids:
                query = query.filter(Conversation.id.in_(conversation_ids))
            
            conversations = await query.all()
            
            export_data = {
                "export_date": datetime.utcnow().isoformat(),
                "format_version": "1.0",
                "user_id": user_id,
                "total_conversations": len(conversations),
                "conversations": []
            }
            
            for conv in conversations:
                conv_data = {
                    "id": conv.id,
                    "title": conv.title,
                    "description": conv.description,
                    "category": conv.category.value if conv.category else None,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                    "messages": []
                }
                
                if include_metadata:
                    conv_data.update({
                        "model_name": conv.model_name,
                        "is_pinned": conv.is_pinned,
                        "is_archived": conv.is_archived,
                        "priority": conv.priority,
                        "color": conv.color,
                        "message_count": conv.message_count,
                        "total_tokens": conv.total_tokens,
                        "tags": [tag.name for tag in conv.tags]
                    })
                
                for msg in conv.messages:
                    msg_data = {
                        "id": msg.id,
                        "role": msg.role.value,
                        "content": msg.content,
                        "message_type": msg.message_type.value,
                        "created_at": msg.created_at.isoformat()
                    }
                    
                    if include_metadata:
                        msg_data.update({
                            "model_name": msg.model_name,
                            "tokens_used": msg.tokens_used,
                            "processing_time": msg.processing_time,
                            "is_important": msg.is_important,
                            "is_favorite": msg.is_favorite
                        })
                    
                    conv_data["messages"].append(msg_data)
                
                export_data["conversations"].append(conv_data)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting conversations: {e}")
            raise


# Global service instances
search_service = ConversationSearchService()
tag_service = TagService()
export_service = ConversationExportService()