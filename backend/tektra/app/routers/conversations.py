"""Conversation management endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.conversation import MessageRole, MessageType
from ..services.conversation_service import conversation_manager

router = APIRouter()


class ConversationCreate(BaseModel):
    """Schema for creating a new conversation."""

    title: Optional[str] = None
    model_name: str = "phi-3-mini"


class ConversationResponse(BaseModel):
    """Schema for conversation response."""

    id: int
    title: str
    model_name: str
    created_at: str
    updated_at: str
    message_count: Optional[int] = None


class MessageCreate(BaseModel):
    """Schema for creating a new message."""

    content: str
    role: str = "user"  # user, assistant, system
    message_type: str = "text"  # text, image, audio, video, action


class MessageResponse(BaseModel):
    """Schema for message response."""

    id: int
    conversation_id: int
    role: str
    content: str
    message_type: str
    created_at: str
    metadata: Dict[str, Any] = {}


class ConversationDetailResponse(BaseModel):
    """Schema for detailed conversation with messages."""

    id: int
    title: str
    model_name: str
    created_at: str
    updated_at: str
    messages: List[MessageResponse]


class ConversationStats(BaseModel):
    """Schema for conversation statistics."""

    total_messages: int
    user_messages: int
    assistant_messages: int
    total_characters: int
    first_message: Optional[str]
    last_message: Optional[str]


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    conversation: ConversationCreate,
    user_id: int = 1,  # TODO: Get from authentication
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Create a new conversation."""
    try:
        new_conversation = await conversation_manager.create_conversation(
            db=db,
            user_id=user_id,
            title=conversation.title,
            model_name=conversation.model_name,
        )

        return ConversationResponse(
            id=new_conversation.id,
            title=new_conversation.title,
            model_name=new_conversation.model_name,
            created_at=new_conversation.created_at.isoformat(),
            updated_at=new_conversation.updated_at.isoformat(),
            message_count=0,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    user_id: int = 1,  # TODO: Get from authentication
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[ConversationResponse]:
    """List user's conversations."""
    try:
        conversations = await conversation_manager.get_user_conversations(
            db=db, user_id=user_id, limit=limit, offset=offset
        )

        # Get message counts
        results = []
        for conv in conversations:
            messages = await conversation_manager.get_conversation_messages(db, conv.id)
            results.append(
                ConversationResponse(
                    id=conv.id,
                    title=conv.title,
                    model_name=conv.model_name,
                    created_at=conv.created_at.isoformat(),
                    updated_at=conv.updated_at.isoformat(),
                    message_count=len(messages),
                )
            )

        return results

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list conversations: {str(e)}"
        )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    user_id: int = 1,  # TODO: Get from authentication
    db: AsyncSession = Depends(get_db),
) -> ConversationDetailResponse:
    """Get a specific conversation with messages."""
    try:
        conversation = await conversation_manager.get_conversation(
            db=db, conversation_id=conversation_id, user_id=user_id
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = await conversation_manager.get_conversation_messages(
            db, conversation_id
        )

        return ConversationDetailResponse(
            id=conversation.id,
            title=conversation.title,
            model_name=conversation.model_name,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            messages=[
                MessageResponse(
                    id=msg.id,
                    conversation_id=msg.conversation_id,
                    role=msg.role.value,
                    content=msg.content,
                    message_type=msg.message_type.value,
                    created_at=msg.created_at.isoformat(),
                    metadata=msg.message_metadata,
                )
                for msg in messages
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get conversation: {str(e)}"
        )


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(
    conversation_id: int,
    message: MessageCreate,
    user_id: int = 1,  # TODO: Get from authentication
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Add a message to a conversation."""
    try:
        # Verify conversation exists and belongs to user
        conversation = await conversation_manager.get_conversation(
            db=db, conversation_id=conversation_id, user_id=user_id
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Convert string enums
        try:
            role = MessageRole(message.role.lower())
            msg_type = MessageType(message.message_type.lower())
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid role or message type: {str(e)}"
            )

        new_message = await conversation_manager.add_message(
            db=db,
            conversation_id=conversation_id,
            role=role,
            content=message.content,
            message_type=msg_type,
        )

        return MessageResponse(
            id=new_message.id,
            conversation_id=new_message.conversation_id,
            role=new_message.role.value,
            content=new_message.content,
            message_type=new_message.message_type.value,
            created_at=new_message.created_at.isoformat(),
            metadata=new_message.message_metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: int,
    user_id: int = 1,  # TODO: Get from authentication
    limit: Optional[int] = Query(None, ge=1, le=500),
    include_system: bool = Query(True),
    db: AsyncSession = Depends(get_db),
) -> List[MessageResponse]:
    """Get messages from a conversation."""
    try:
        # Verify conversation exists and belongs to user
        conversation = await conversation_manager.get_conversation(
            db=db, conversation_id=conversation_id, user_id=user_id
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = await conversation_manager.get_conversation_messages(
            db=db,
            conversation_id=conversation_id,
            limit=limit,
            include_system=include_system,
        )

        return [
            MessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role.value,
                content=msg.content,
                message_type=msg.message_type.value,
                created_at=msg.created_at.isoformat(),
                metadata=msg.message_metadata,
            )
            for msg in messages
        ]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")


@router.put("/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: int,
    title: str,
    user_id: int = 1,  # TODO: Get from authentication
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Update conversation title."""
    try:
        success = await conversation_manager.update_conversation_title(
            db=db, conversation_id=conversation_id, title=title, user_id=user_id
        )

        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": "Title updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    user_id: int = 1,  # TODO: Get from authentication
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Delete a conversation."""
    try:
        success = await conversation_manager.delete_conversation(
            db=db, conversation_id=conversation_id, user_id=user_id
        )

        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": "Conversation deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete conversation: {str(e)}"
        )


@router.get("/{conversation_id}/stats", response_model=ConversationStats)
async def get_conversation_stats(
    conversation_id: int,
    user_id: int = 1,  # TODO: Get from authentication
    db: AsyncSession = Depends(get_db),
) -> ConversationStats:
    """Get conversation statistics."""
    try:
        # Verify conversation exists and belongs to user
        conversation = await conversation_manager.get_conversation(
            db=db, conversation_id=conversation_id, user_id=user_id
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        stats = await conversation_manager.get_conversation_stats(db, conversation_id)

        return ConversationStats(**stats)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/search/{query}", response_model=List[ConversationResponse])
async def search_conversations(
    query: str,
    user_id: int = 1,  # TODO: Get from authentication
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[ConversationResponse]:
    """Search conversations."""
    try:
        conversations = await conversation_manager.search_conversations(
            db=db, user_id=user_id, query=query, limit=limit
        )

        return [
            ConversationResponse(
                id=conv.id,
                title=conv.title,
                model_name=conv.model_name,
                created_at=conv.created_at.isoformat(),
                updated_at=conv.updated_at.isoformat(),
            )
            for conv in conversations
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to search conversations: {str(e)}"
        )


@router.get("/{conversation_id}/context")
async def get_ai_context(
    conversation_id: int,
    user_id: int = 1,  # TODO: Get from authentication
    max_messages: Optional[int] = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get conversation context formatted for AI."""
    try:
        # Verify conversation exists and belongs to user
        conversation = await conversation_manager.get_conversation(
            db=db, conversation_id=conversation_id, user_id=user_id
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        context = await conversation_manager.get_context_for_ai(
            db=db, conversation_id=conversation_id, max_messages=max_messages
        )

        return {
            "conversation_id": conversation_id,
            "model_name": conversation.model_name,
            "context": context,
            "message_count": len(context),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")
