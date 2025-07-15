"""
MemOS Integration for Tektra Conversational Memory

This module integrates MemTensor's MemOS (https://github.com/MemTensor/MemOS) 
for sophisticated conversational memory capabilities.

MemOS provides:
- Persistent memory across conversations
- Semantic search and retrieval
- Memory consolidation and forgetting
- Context-aware memory formation
- Efficient vector-based storage
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from memos import MemOS, Memory, MemoryType as MemOSMemoryType
    from memos.config import MemOSConfig
    MEMOS_AVAILABLE = True
    logger.info("MemOS successfully imported")
except ImportError as e:
    logger.warning(f"MemOS not available: {e}. Using fallback memory system.")
    MEMOS_AVAILABLE = False
    # Create mock classes for graceful degradation
    class MemOS:
        def __init__(self, *args, **kwargs):
            pass
    class Memory:
        def __init__(self, *args, **kwargs):
            pass
    class MemOSMemoryType:
        EPISODIC = "episodic"
        SEMANTIC = "semantic"
        WORKING = "working"
    class MemOSConfig:
        def __init__(self, *args, **kwargs):
            pass


class FallbackMemorySystem:
    """
    Fallback memory system when MemOS is not available.
    
    Provides basic conversation memory functionality using local JSON storage.
    """
    
    def __init__(self, memory_dir: Path):
        """Initialize fallback memory system."""
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_file = memory_dir / "conversations.json"
        self.memories_file = memory_dir / "memories.json"
        self.conversations = self._load_conversations()
        self.memories = self._load_memories()
        
    def _load_conversations(self) -> List[Dict]:
        """Load conversation history from file."""
        try:
            if self.conversations_file.exists():
                with open(self.conversations_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
        return []
    
    def _save_conversations(self):
        """Save conversation history to file."""
        try:
            with open(self.conversations_file, 'w') as f:
                json.dump(self.conversations, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
    
    def _load_memories(self) -> Dict[str, List[Dict]]:
        """Load memory index from file."""
        try:
            if self.memories_file.exists():
                with open(self.memories_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
        return {}
    
    def _save_memories(self):
        """Save memory index to file."""
        try:
            with open(self.memories_file, 'w') as f:
                json.dump(self.memories, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    async def store_conversation_turn(self, user_message: str, assistant_response: str, metadata: Dict = None):
        """Store a conversation turn."""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "metadata": metadata or {}
        }
        self.conversations.append(turn)
        
        # Keep only last 100 turns to prevent unlimited growth
        if len(self.conversations) > 100:
            self.conversations = self.conversations[-100:]
        
        self._save_conversations()
        
        # Index key phrases for retrieval
        await self._index_conversation_turn(turn)
    
    async def _index_conversation_turn(self, turn: Dict):
        """Create simple keyword index for conversation turn."""
        keywords = self._extract_keywords(turn["user_message"]) + self._extract_keywords(turn["assistant_response"])
        
        for keyword in keywords:
            if keyword not in self.memories:
                self.memories[keyword] = []
            
            self.memories[keyword].append({
                "timestamp": turn["timestamp"],
                "user_message": turn["user_message"][:100],
                "assistant_response": turn["assistant_response"][:100],
                "relevance": 1.0
            })
            
            # Keep only last 10 memories per keyword
            if len(self.memories[keyword]) > 10:
                self.memories[keyword] = self.memories[keyword][-10:]
        
        self._save_memories()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text."""
        # Simple keyword extraction - split on spaces and filter
        words = text.lower().split()
        keywords = []
        
        stopwords = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'what', 'when', 'where', 'the', 'and', 'for', 'you', 'can', 'are', 'not', 'but', 'all', 'one', 'her', 'his', 'him', 'she', 'our', 'out', 'may', 'had', 'how', 'who', 'was', 'its', 'did', 'get', 'now', 'use', 'man', 'new', 'way', 'see', 'two', 'day', 'any', 'old', 'say'}
        
        for word in words:
            # Remove punctuation and filter short words
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 3 and clean_word not in stopwords:
                keywords.append(clean_word)
        
        return keywords[:5]  # Return top 5 keywords
    
    async def get_relevant_context(self, query: str, max_memories: int = 5, include_recent: bool = True) -> str:
        """Get relevant context for a query."""
        keywords = self._extract_keywords(query)
        relevant_memories = []
        
        # Find memories matching keywords
        for keyword in keywords:
            if keyword in self.memories:
                relevant_memories.extend(self.memories[keyword])
        
        # Add recent conversations if requested
        if include_recent and self.conversations:
            for turn in self.conversations[-3:]:  # Last 3 turns
                relevant_memories.append({
                    "timestamp": turn["timestamp"],
                    "user_message": turn["user_message"][:100],
                    "assistant_response": turn["assistant_response"][:100],
                    "relevance": 0.8
                })
        
        # Sort by relevance and timestamp
        relevant_memories.sort(key=lambda x: (x["relevance"], x["timestamp"]), reverse=True)
        relevant_memories = relevant_memories[:max_memories]
        
        if not relevant_memories:
            return ""
        
        # Format context
        context_parts = []
        for memory in relevant_memories:
            context_parts.append(f"Previous: {memory['user_message']}")
            context_parts.append(f"Response: {memory['assistant_response']}")
        
        return "\n".join(context_parts)


class TektraMemOSIntegration:
    """
    Integration layer between Tektra and MemOS for conversational memory.
    
    This class provides a simplified interface for using MemOS in Tektra
    conversations while maintaining backward compatibility.
    """

    def __init__(self, memory_dir: Path, user_id: str = "default_user"):
        """
        Initialize MemOS integration.
        
        Args:
            memory_dir: Directory for memory storage
            user_id: Unique identifier for the user
        """
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        
        self.memos_available = MEMOS_AVAILABLE
        self.memos = None
        
        if self.memos_available:
            try:
                self._initialize_memos()
            except Exception as e:
                logger.error(f"Failed to initialize MemOS: {e}. Using fallback system.")
                self.memos_available = False
                self._initialize_fallback()
        else:
            logger.warning("MemOS not available, using fallback memory system")
            self._initialize_fallback()

    def _initialize_fallback(self):
        """Initialize fallback memory system."""
        self.fallback_memory = FallbackMemorySystem(self.memory_dir)

    def _initialize_memos(self):
        """Initialize MemOS with Tektra-specific configuration."""
        try:
            # Configure MemOS for conversational use
            config = MemOSConfig(
                storage_path=str(self.memory_dir / "memos_storage"),
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight model
                memory_consolidation_threshold=100,  # Consolidate after 100 memories
                max_working_memory_size=50,  # Keep recent context
                semantic_similarity_threshold=0.7,  # For memory retrieval
                enable_forgetting=True,  # Allow natural forgetting
                forgetting_curve_factor=0.9  # Ebbinghaus forgetting curve
            )
            
            # Initialize MemOS
            self.memos = MemOS(
                user_id=self.user_id,
                config=config
            )
            
            logger.info(f"MemOS initialized for user: {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MemOS: {e}")
            self.memos_available = False
            self._initialize_fallback()

    def _initialize_fallback(self):
        """Initialize fallback memory system when MemOS is not available."""
        self.fallback_memories = []
        self.fallback_max_size = 1000
        logger.info("Initialized fallback memory system")

    async def add_conversation_turn(
        self,
        user_message: str,
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a conversation turn to memory.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            context: Optional context information
            
        Returns:
            bool: True if successfully added
        """
        if self.memos_available and self.memos:
            return await self._add_to_memos(user_message, assistant_response, context)
        else:
            return await self._add_to_fallback(user_message, assistant_response, context)

    async def _add_to_memos(
        self,
        user_message: str,
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add conversation turn to MemOS."""
        try:
            # Create episodic memory for the conversation turn
            conversation_memory = Memory(
                content=f"User: {user_message}\nAssistant: {assistant_response}",
                memory_type=MemOSMemoryType.EPISODIC,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "user_message": user_message,
                    "assistant_response": assistant_response,
                    "context": context or {},
                    "conversation_turn": True
                },
                importance=0.5  # Default importance
            )
            
            # Store in MemOS
            await self.memos.store_memory(conversation_memory)
            
            # Extract and store semantic information
            await self._extract_semantic_memories(user_message, context)
            
            logger.debug(f"Added conversation turn to MemOS: {user_message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add conversation to MemOS: {e}")
            return False

    async def _add_to_fallback(
        self,
        user_message: str,
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add conversation turn to fallback memory."""
        try:
            await self.fallback_memory.store_conversation_turn(
                user_message, assistant_response, context
            )
            logger.debug(f"Added conversation turn to fallback memory: {user_message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add conversation to fallback memory: {e}")
            return False

    async def _extract_semantic_memories(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Extract semantic information from user message."""
        if not (self.memos_available and self.memos):
            return
        
        try:
            # Simple patterns for extracting facts and preferences
            message_lower = user_message.lower()
            
            # Extract preferences
            if "i like" in message_lower or "i love" in message_lower:
                semantic_memory = Memory(
                    content=f"User preference: {user_message}",
                    memory_type=MemOSMemoryType.SEMANTIC,
                    metadata={
                        "type": "preference",
                        "polarity": "positive",
                        "extracted_from": user_message,
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=0.8  # Preferences are important
                )
                await self.memos.store_memory(semantic_memory)
            
            # Extract dislikes
            elif "i don't like" in message_lower or "i hate" in message_lower:
                semantic_memory = Memory(
                    content=f"User dislike: {user_message}",
                    memory_type=MemOSMemoryType.SEMANTIC,
                    metadata={
                        "type": "preference",
                        "polarity": "negative",
                        "extracted_from": user_message,
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=0.8
                )
                await self.memos.store_memory(semantic_memory)
            
            # Extract personal information
            personal_patterns = [
                ("my name is", "name"),
                ("i work as", "occupation"),
                ("i am a", "role"),
                ("i live in", "location"),
                ("i study", "education")
            ]
            
            for pattern, info_type in personal_patterns:
                if pattern in message_lower:
                    semantic_memory = Memory(
                        content=f"User {info_type}: {user_message}",
                        memory_type=MemOSMemoryType.SEMANTIC,
                        metadata={
                            "type": "personal_info",
                            "info_type": info_type,
                            "extracted_from": user_message,
                            "timestamp": datetime.now().isoformat()
                        },
                        importance=0.9  # Personal info is very important
                    )
                    await self.memos.store_memory(semantic_memory)
                    break
                    
        except Exception as e:
            logger.warning(f"Failed to extract semantic memories: {e}")

    async def get_relevant_context(
        self,
        query: str,
        max_memories: int = 5,
        include_recent: bool = True
    ) -> str:
        """
        Get relevant context for a query.
        
        Args:
            query: Current user query
            max_memories: Maximum number of memories to retrieve
            include_recent: Include recent conversation context
            
        Returns:
            str: Formatted context string
        """
        if self.memos_available and self.memos:
            return await self._get_memos_context(query, max_memories, include_recent)
        else:
            return await self._get_fallback_context(query, max_memories, include_recent)

    async def _get_memos_context(
        self,
        query: str,
        max_memories: int = 5,
        include_recent: bool = True
    ) -> str:
        """Get context from MemOS."""
        try:
            context_parts = []
            
            # Get semantically relevant memories
            relevant_memories = await self.memos.retrieve_memories(
                query=query,
                memory_types=[MemOSMemoryType.SEMANTIC, MemOSMemoryType.EPISODIC],
                limit=max_memories,
                min_similarity=0.6
            )
            
            if relevant_memories:
                context_parts.append("## Relevant Context:")
                for memory in relevant_memories:
                    # Format memory content
                    if memory.metadata.get("type") == "preference":
                        polarity = memory.metadata.get("polarity", "")
                        context_parts.append(f"- User {polarity} preference: {memory.content}")
                    elif memory.metadata.get("type") == "personal_info":
                        info_type = memory.metadata.get("info_type", "")
                        context_parts.append(f"- User {info_type}: {memory.content}")
                    else:
                        context_parts.append(f"- {memory.content}")
            
            # Get recent working memory if requested
            if include_recent:
                recent_memories = await self.memos.get_working_memory()
                if recent_memories:
                    context_parts.append("\n## Recent Context:")
                    for memory in recent_memories[-3:]:  # Last 3 entries
                        if memory.metadata.get("conversation_turn"):
                            user_msg = memory.metadata.get("user_message", "")
                            if user_msg:
                                context_parts.append(f"- User mentioned: {user_msg}")
            
            return "\n".join(context_parts) + "\n" if context_parts else ""
            
        except Exception as e:
            logger.error(f"Failed to get MemOS context: {e}")
            return ""

    async def _get_fallback_context(
        self,
        query: str,
        max_memories: int = 5,
        include_recent: bool = True
    ) -> str:
        """Get context from fallback memory."""
        try:
            return await self.fallback_memory.get_relevant_context(
                query, max_memories, include_recent
            )
        except Exception as e:
            logger.error(f"Failed to get fallback context: {e}")
            return ""

    async def get_user_preferences(self) -> Dict[str, List[str]]:
        """
        Get user preferences organized by type.
        
        Returns:
            dict: Dictionary with 'likes' and 'dislikes' lists
        """
        if self.memos_available and self.memos:
            return await self._get_memos_preferences()
        else:
            return await self._get_fallback_preferences()

    async def _get_memos_preferences(self) -> Dict[str, List[str]]:
        """Get preferences from MemOS."""
        try:
            preferences = {"likes": [], "dislikes": []}
            
            # Query for preference memories
            preference_memories = await self.memos.retrieve_memories(
                query="preference",
                memory_types=[MemOSMemoryType.SEMANTIC],
                limit=20,
                min_similarity=0.5
            )
            
            for memory in preference_memories:
                if memory.metadata.get("type") == "preference":
                    polarity = memory.metadata.get("polarity", "")
                    content = memory.content
                    
                    if polarity == "positive":
                        preferences["likes"].append(content)
                    elif polarity == "negative":
                        preferences["dislikes"].append(content)
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get MemOS preferences: {e}")
            return {"likes": [], "dislikes": []}

    async def _get_fallback_preferences(self) -> Dict[str, List[str]]:
        """Get preferences from fallback memory."""
        preferences = {"likes": [], "dislikes": []}
        
        for memory in self.fallback_memories:
            user_msg = memory.get("user_message", "").lower()
            
            if "i like" in user_msg or "i love" in user_msg:
                preferences["likes"].append(user_msg)
            elif "i don't like" in user_msg or "i hate" in user_msg:
                preferences["dislikes"].append(user_msg)
        
        return preferences

    async def consolidate_memories(self) -> bool:
        """
        Trigger memory consolidation process.
        
        Returns:
            bool: True if consolidation was performed
        """
        if self.memos_available and self.memos:
            try:
                await self.memos.consolidate_memories()
                logger.info("Memory consolidation completed")
                return True
            except Exception as e:
                logger.error(f"Memory consolidation failed: {e}")
                return False
        else:
            # For fallback, just trim old memories
            if len(self.fallback_memories) > self.fallback_max_size:
                self.fallback_memories = self.fallback_memories[-self.fallback_max_size:]
                logger.info("Fallback memory trimmed")
            return True

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if self.memos_available and self.memos:
            try:
                stats = self.memos.get_memory_stats()
                stats["system"] = "MemOS"
                stats["user_id"] = self.user_id
                return stats
            except Exception as e:
                logger.error(f"Failed to get MemOS stats: {e}")
                return {"system": "MemOS (error)", "error": str(e)}
        else:
            return {
                "system": "Fallback",
                "total_memories": len(self.fallback_memories),
                "max_size": self.fallback_max_size,
                "user_id": self.user_id
            }

    async def clear_memories(self, confirm: bool = False) -> bool:
        """
        Clear all memories (use with caution).
        
        Args:
            confirm: Must be True to actually clear memories
            
        Returns:
            bool: True if memories were cleared
        """
        if not confirm:
            logger.warning("Memory clear attempted without confirmation")
            return False
        
        if self.memos_available and self.memos:
            try:
                await self.memos.clear_all_memories()
                logger.warning("All MemOS memories cleared")
                return True
            except Exception as e:
                logger.error(f"Failed to clear MemOS memories: {e}")
                return False
        else:
            self.fallback_memories.clear()
            logger.warning("All fallback memories cleared")
            return True