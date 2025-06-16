"""
Enhanced Conversation Management Endpoints with Search and Organization.

Advanced features for conversation search, tagging, categorization, and analytics.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.conversation import (
    Conversation,
    ConversationCategory,
    Message,
    MessageRole,
    MessageType,
    Tag,
)
from ..services.conversation_search_service import (
    export_service,
    search_service,
    tag_service,
)
from ..services.conversation_service import conversation_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# Enhanced Pydantic Models


class ConversationSearchRequest(BaseModel):
    """Request model for conversation search."""

    query: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[ConversationCategory] = None
    model_name: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    is_pinned: Optional[bool] = None
    is_archived: Optional[bool] = None
    min_messages: Optional[int] = None
    max_messages: Optional[int] = None
    sort_by: str = Field(
        default="updated_at",
        pattern="^(created_at|updated_at|title|message_count|priority)$",
    )
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class MessageSearchRequest(BaseModel):
    """Request model for message search."""

    query: str
    conversation_id: Optional[int] = None
    role: Optional[MessageRole] = None
    message_type: Optional[MessageType] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    is_favorite: Optional[bool] = None
    is_important: Optional[bool] = None
    limit: int = Field(default=100, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class ConversationUpdateRequest(BaseModel):
    """Request model for updating conversation metadata."""

    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[ConversationCategory] = None
    is_pinned: Optional[bool] = None
    is_archived: Optional[bool] = None
    priority: Optional[int] = Field(None, ge=0, le=10)
    color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")


class TagCreateRequest(BaseModel):
    """Request model for creating tags."""

    name: str = Field(..., min_length=1, max_length=50)
    color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")
    description: Optional[str] = Field(None, max_length=255)


class TagAssignRequest(BaseModel):
    """Request model for assigning tags to conversations."""

    conversation_id: int
    tag_names: List[str] = Field(..., min_items=1)


class MessageUpdateRequest(BaseModel):
    """Request model for updating message metadata."""

    is_important: Optional[bool] = None
    is_favorite: Optional[bool] = None


class ConversationExportRequest(BaseModel):
    """Request model for exporting conversations."""

    conversation_ids: Optional[List[int]] = None
    format: str = Field(default="json", pattern="^(json|csv|markdown)$")
    include_metadata: bool = True


class ConversationResponse(BaseModel):
    """Enhanced conversation response model."""

    id: int
    title: Optional[str]
    description: Optional[str]
    category: ConversationCategory
    model_name: Optional[str]
    is_active: bool
    is_pinned: bool
    is_archived: bool
    priority: int
    color: Optional[str]
    message_count: int
    total_tokens: int
    avg_response_time: Optional[float]
    last_message_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]
    tags: List[str] = []


class MessageResponse(BaseModel):
    """Enhanced message response model."""

    id: int
    conversation_id: int
    role: MessageRole
    message_type: MessageType
    content: str
    model_name: Optional[str]
    tokens_used: Optional[int]
    processing_time: Optional[int]
    is_important: bool
    is_favorite: bool
    created_at: datetime
    updated_at: Optional[datetime]


class TagResponse(BaseModel):
    """Tag response model."""

    id: int
    name: str
    color: Optional[str]
    description: Optional[str]
    usage_count: int
    created_at: datetime


# Search and Discovery Endpoints


@router.post("/search")
async def search_conversations(
    request: ConversationSearchRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Advanced conversation search with multiple filters."""
    try:
        conversations, total_count = await search_service.search_conversations(
            db=db,
            user_id=user_id,
            query=request.query,
            tags=request.tags,
            category=request.category,
            model_name=request.model_name,
            date_from=request.date_from,
            date_to=request.date_to,
            is_pinned=request.is_pinned,
            is_archived=request.is_archived,
            min_messages=request.min_messages,
            max_messages=request.max_messages,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
            limit=request.limit,
            offset=request.offset,
        )

        # Convert to response format
        conversation_responses = []
        for conv in conversations:
            response = ConversationResponse(
                id=conv.id,
                title=conv.title,
                description=conv.description,
                category=conv.category,
                model_name=conv.model_name,
                is_active=conv.is_active,
                is_pinned=conv.is_pinned,
                is_archived=conv.is_archived,
                priority=conv.priority,
                color=conv.color,
                message_count=conv.message_count,
                total_tokens=conv.total_tokens,
                avg_response_time=conv.avg_response_time,
                last_message_at=conv.last_message_at,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                tags=[tag.name for tag in conv.tags],
            )
            conversation_responses.append(response)

        return {
            "status": "success",
            "conversations": conversation_responses,
            "pagination": {
                "total": total_count,
                "limit": request.limit,
                "offset": request.offset,
                "has_more": (request.offset + request.limit) < total_count,
            },
        }

    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/search")
async def search_messages(
    request: MessageSearchRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Search messages across conversations."""
    try:
        messages, total_count = await search_service.search_messages(
            db=db,
            user_id=user_id,
            query=request.query,
            conversation_id=request.conversation_id,
            role=request.role,
            message_type=request.message_type,
            date_from=request.date_from,
            date_to=request.date_to,
            is_favorite=request.is_favorite,
            is_important=request.is_important,
            limit=request.limit,
            offset=request.offset,
        )

        # Convert to response format
        message_responses = []
        for msg in messages:
            response = MessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role,
                message_type=msg.message_type,
                content=msg.content,
                model_name=msg.model_name,
                tokens_used=msg.tokens_used,
                processing_time=msg.processing_time,
                is_important=msg.is_important,
                is_favorite=msg.is_favorite,
                created_at=msg.created_at,
                updated_at=msg.updated_at,
            )
            message_responses.append(response)

        return {
            "status": "success",
            "messages": message_responses,
            "pagination": {
                "total": total_count,
                "limit": request.limit,
                "offset": request.offset,
                "has_more": (request.offset + request.limit) < total_count,
            },
        }

    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Organization and Metadata Endpoints


@router.put("/{conversation_id}/metadata")
async def update_conversation_metadata(
    conversation_id: int,
    request: ConversationUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Update conversation metadata (title, category, pinning, etc.)."""
    try:
        # Get conversation
        conversation = (
            await db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .first()
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update fields
        update_data = request.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(conversation, field, value)

        conversation.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(conversation)

        return {
            "status": "success",
            "message": "Conversation metadata updated",
            "conversation_id": conversation_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating conversation metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/messages/{message_id}/metadata")
async def update_message_metadata(
    message_id: int,
    request: MessageUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Update message metadata (favorite, important, etc.)."""
    try:
        # Get message through conversation ownership
        message = (
            await db.query(Message)
            .join(Conversation)
            .filter(Message.id == message_id, Conversation.user_id == user_id)
            .first()
        )

        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        # Update fields
        update_data = request.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(message, field, value)

        message.updated_at = datetime.utcnow()

        await db.commit()

        return {
            "status": "success",
            "message": "Message metadata updated",
            "message_id": message_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating message metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Tag Management Endpoints


@router.get("/tags")
async def get_user_tags(
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
    include_usage: bool = True,
) -> Dict[str, Any]:
    """Get all tags for the current user."""
    try:
        tags = await tag_service.get_user_tags(db, user_id, include_usage)

        tag_responses = []
        for tag in tags:
            response = TagResponse(
                id=tag.id,
                name=tag.name,
                color=tag.color,
                description=tag.description,
                usage_count=tag.usage_count,
                created_at=tag.created_at,
            )
            tag_responses.append(response)

        return {"status": "success", "tags": tag_responses, "total": len(tag_responses)}

    except Exception as e:
        logger.error(f"Error getting user tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tags")
async def create_tag(
    request: TagCreateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Create a new tag."""
    try:
        tag = await tag_service.create_tag(
            db=db,
            user_id=user_id,
            name=request.name,
            color=request.color,
            description=request.description,
        )

        response = TagResponse(
            id=tag.id,
            name=tag.name,
            color=tag.color,
            description=tag.description,
            usage_count=tag.usage_count,
            created_at=tag.created_at,
        )

        return {
            "status": "success",
            "message": "Tag created successfully",
            "tag": response,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating tag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tags/assign")
async def assign_tags_to_conversation(
    request: TagAssignRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Assign tags to a conversation."""
    try:
        success = await tag_service.add_tags_to_conversation(
            db=db,
            conversation_id=request.conversation_id,
            tag_names=request.tag_names,
            user_id=user_id,
        )

        if success:
            return {
                "status": "success",
                "message": f"Tags assigned to conversation {request.conversation_id}",
                "conversation_id": request.conversation_id,
                "tags": request.tag_names,
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to assign tags")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error assigning tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tags/{tag_id}")
async def delete_tag(
    tag_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Delete a tag."""
    try:
        success = await tag_service.delete_tag(db, tag_id, user_id)

        if success:
            return {
                "status": "success",
                "message": "Tag deleted successfully",
                "tag_id": tag_id,
            }
        else:
            raise HTTPException(status_code=404, detail="Tag not found")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting tag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics and Export Endpoints


@router.get("/analytics")
async def get_conversation_analytics(
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
    conversation_id: Optional[int] = None,
    days: int = Query(default=30, ge=1, le=365),
) -> Dict[str, Any]:
    """Get conversation analytics and insights."""
    try:
        analytics = await search_service.get_conversation_analytics(
            db=db, user_id=user_id, conversation_id=conversation_id, days=days
        )

        return {"status": "success", "analytics": analytics}

    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_conversations(
    request: ConversationExportRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Export conversations to various formats."""
    try:
        export_data = await export_service.export_conversations(
            db=db,
            user_id=user_id,
            conversation_ids=request.conversation_ids,
            format=request.format,
            include_metadata=request.include_metadata,
        )

        return {"status": "success", "export_data": export_data}

    except Exception as e:
        logger.error(f"Error exporting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Bulk Operations


@router.post("/bulk/archive")
async def bulk_archive_conversations(
    conversation_ids: List[int],
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
) -> Dict[str, Any]:
    """Archive multiple conversations."""
    try:
        updated_count = (
            await db.query(Conversation)
            .filter(
                Conversation.id.in_(conversation_ids), Conversation.user_id == user_id
            )
            .update(
                {"is_archived": True, "updated_at": datetime.utcnow()},
                synchronize_session=False,
            )
        )

        await db.commit()

        return {
            "status": "success",
            "message": f"Archived {updated_count} conversations",
            "updated_count": updated_count,
        }

    except Exception as e:
        await db.rollback()
        logger.error(f"Error bulk archiving conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/delete")
async def bulk_delete_conversations(
    conversation_ids: List[int],
    db: AsyncSession = Depends(get_db),
    user_id: int = 1,  # TODO: Get from authentication
    permanent: bool = False,
) -> Dict[str, Any]:
    """Delete multiple conversations (soft delete by default)."""
    try:
        if permanent:
            # Permanent deletion
            deleted_count = (
                await db.query(Conversation)
                .filter(
                    Conversation.id.in_(conversation_ids),
                    Conversation.user_id == user_id,
                )
                .delete(synchronize_session=False)
            )
        else:
            # Soft delete (mark as inactive)
            deleted_count = (
                await db.query(Conversation)
                .filter(
                    Conversation.id.in_(conversation_ids),
                    Conversation.user_id == user_id,
                )
                .update(
                    {"is_active": False, "updated_at": datetime.utcnow()},
                    synchronize_session=False,
                )
            )

        await db.commit()

        return {
            "status": "success",
            "message": f"Deleted {deleted_count} conversations",
            "deleted_count": deleted_count,
            "permanent": permanent,
        }

    except Exception as e:
        await db.rollback()
        logger.error(f"Error bulk deleting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
