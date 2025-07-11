"""
Vector Database

This module provides vector database functionality for Tektra AI Assistant,
including embeddings storage, similarity search, and semantic retrieval.
"""

import json
import sqlite3
import uuid
from typing import Any

import numpy as np
from loguru import logger

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - using mock vector database")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available - using mock embeddings")


class VectorDatabase:
    """Vector database for semantic search and embeddings storage."""

    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector database.

        Args:
            db_path: Path to the database file
            embedding_model: Name of the embedding model to use
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.vector_index = None
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2

        # Initialize database
        self.init_database()

        # Initialize embedding model
        self.init_embedding_model()

        # Initialize vector index
        self.init_vector_index()

        logger.info(f"Vector database initialized at {db_path}")

    def init_database(self):
        """Initialize the SQLite database for metadata storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        vector_index INTEGER,
                        content_hash TEXT,
                        source_type TEXT,
                        source_id TEXT
                    )
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_hash
                    ON embeddings(content_hash)
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_source
                    ON embeddings(source_type, source_id)
                """
                )

                conn.commit()
                logger.debug("Vector database schema initialized")

        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise

    def init_embedding_model(self):
        """Initialize the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Sentence transformers not available - using mock embeddings"
            )
            return

        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(
                f"Embedding model '{self.embedding_model_name}' loaded (dimension: {self.dimension})"
            )

        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None

    def init_vector_index(self):
        """Initialize the FAISS vector index."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - using mock vector index")
            return

        try:
            # Create FAISS index
            self.vector_index = faiss.IndexFlatL2(self.dimension)

            # Load existing vectors if any
            self.load_existing_vectors()

            logger.info(
                f"Vector index initialized with {self.vector_index.ntotal} vectors"
            )

        except Exception as e:
            logger.error(f"Error initializing vector index: {e}")
            self.vector_index = None

    def load_existing_vectors(self):
        """Load existing vectors from database into FAISS index."""
        if not self.vector_index:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, content FROM embeddings
                    ORDER BY vector_index
                """
                )

                contents = []
                for row in cursor:
                    contents.append(row[1])  # content

                if contents:
                    vectors = self.generate_embeddings(contents)
                    if vectors is not None:
                        self.vector_index.add(vectors)
                        logger.info(f"Loaded {len(contents)} existing vectors")

        except Exception as e:
            logger.error(f"Error loading existing vectors: {e}")

    def generate_embeddings(self, texts: list[str]) -> np.ndarray | None:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings or None if model not available
        """
        if not self.embedding_model:
            # Return mock embeddings
            return np.random.rand(len(texts), self.dimension).astype(np.float32)

        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    def add_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
    ) -> str:
        """
        Add text to the vector database.

        Args:
            text: Text content to add
            metadata: Optional metadata dictionary
            source_type: Type of source (e.g., "conversation", "document")
            source_id: ID of the source

        Returns:
            ID of the added text
        """
        try:
            # Generate ID and hash
            text_id = str(uuid.uuid4())
            content_hash = self._get_content_hash(text)

            # Check if content already exists
            if self.content_exists(content_hash):
                logger.debug(f"Content already exists with hash: {content_hash}")
                return self.get_id_by_hash(content_hash)

            # Generate embedding
            embedding = self.generate_embeddings([text])
            if embedding is None:
                raise ValueError("Failed to generate embedding")

            # Get vector index
            vector_index = self.vector_index.ntotal if self.vector_index else 0

            # Add to vector index
            if self.vector_index:
                self.vector_index.add(embedding)

            # Add to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO embeddings
                    (id, content, metadata, vector_index, content_hash, source_type, source_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        text_id,
                        text,
                        json.dumps(metadata) if metadata else None,
                        vector_index,
                        content_hash,
                        source_type,
                        source_id,
                    ),
                )
                conn.commit()

            logger.debug(f"Added text to vector database: {text_id}")
            return text_id

        except Exception as e:
            logger.error(f"Error adding text to vector database: {e}")
            raise

    def search_similar(
        self, query: str, k: int = 5, score_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Search for similar texts in the database.

        Args:
            query: Query text to search for
            k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of similar texts with metadata
        """
        try:
            if not self.vector_index or self.vector_index.ntotal == 0:
                logger.warning("Vector index empty or not available")
                return []

            # Generate query embedding
            query_embedding = self.generate_embeddings([query])
            if query_embedding is None:
                return []

            # Search in vector index
            distances, indices = self.vector_index.search(query_embedding, k)

            # Get results from database
            results = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                for _i, (distance, index) in enumerate(zip(distances[0], indices[0], strict=False)):
                    if index == -1:  # Invalid index
                        continue

                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity_score = max(0, 1 - (distance / 2))  # Normalize to 0-1

                    if similarity_score < score_threshold:
                        continue

                    cursor = conn.execute(
                        """
                        SELECT id, content, metadata, created_at, source_type, source_id
                        FROM embeddings
                        WHERE vector_index = ?
                    """,
                        (int(index),),
                    )

                    row = cursor.fetchone()
                    if row:
                        result = {
                            "id": row["id"],
                            "content": row["content"],
                            "metadata": (
                                json.loads(row["metadata"]) if row["metadata"] else {}
                            ),
                            "created_at": row["created_at"],
                            "source_type": row["source_type"],
                            "source_id": row["source_id"],
                            "similarity_score": similarity_score,
                            "distance": float(distance),
                        }
                        results.append(result)

            # Sort by similarity score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            logger.debug(
                f"Found {len(results)} similar texts for query: {query[:50]}..."
            )
            return results

        except Exception as e:
            logger.error(f"Error searching similar texts: {e}")
            return []

    def get_by_id(self, text_id: str) -> dict[str, Any] | None:
        """
        Get text by ID.

        Args:
            text_id: ID of the text

        Returns:
            Text data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT id, content, metadata, created_at, source_type, source_id
                    FROM embeddings
                    WHERE id = ?
                """,
                    (text_id,),
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                        "created_at": row["created_at"],
                        "source_type": row["source_type"],
                        "source_id": row["source_id"],
                    }

                return None

        except Exception as e:
            logger.error(f"Error getting text by ID: {e}")
            return None

    def delete_by_id(self, text_id: str) -> bool:
        """
        Delete text by ID.

        Args:
            text_id: ID of the text to delete

        Returns:
            True if deleted successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM embeddings WHERE id = ?
                """,
                    (text_id,),
                )

                deleted = cursor.rowcount > 0
                conn.commit()

                if deleted:
                    logger.info(f"Deleted text from vector database: {text_id}")
                    # Note: We don't remove from FAISS index as it's not trivial
                    # The index should be rebuilt periodically

                return deleted

        except Exception as e:
            logger.error(f"Error deleting text by ID: {e}")
            return False

    def delete_by_source(self, source_type: str, source_id: str) -> int:
        """
        Delete texts by source.

        Args:
            source_type: Type of source
            source_id: ID of the source

        Returns:
            Number of deleted texts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM embeddings
                    WHERE source_type = ? AND source_id = ?
                """,
                    (source_type, source_id),
                )

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(
                    f"Deleted {deleted_count} texts from source {source_type}:{source_id}"
                )
                return deleted_count

        except Exception as e:
            logger.error(f"Error deleting texts by source: {e}")
            return 0

    def content_exists(self, content_hash: str) -> bool:
        """Check if content with given hash exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT 1 FROM embeddings WHERE content_hash = ?
                """,
                    (content_hash,),
                )

                return cursor.fetchone() is not None

        except Exception as e:
            logger.error(f"Error checking content existence: {e}")
            return False

    def get_id_by_hash(self, content_hash: str) -> str | None:
        """Get text ID by content hash."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id FROM embeddings WHERE content_hash = ?
                """,
                    (content_hash,),
                )

                row = cursor.fetchone()
                return row[0] if row else None

        except Exception as e:
            logger.error(f"Error getting ID by hash: {e}")
            return None

    def rebuild_index(self):
        """Rebuild the FAISS index from database."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - cannot rebuild index")
            return

        try:
            # Create new index
            self.vector_index = faiss.IndexFlatL2(self.dimension)

            # Load all texts and regenerate embeddings
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, content FROM embeddings
                    ORDER BY created_at
                """
                )

                contents = []
                for row in cursor:
                    contents.append(row[1])  # content

                if contents:
                    vectors = self.generate_embeddings(contents)
                    if vectors is not None:
                        self.vector_index.add(vectors)

                        # Update vector indices in database
                        for i, row in enumerate(cursor):
                            conn.execute(
                                """
                                UPDATE embeddings
                                SET vector_index = ?
                                WHERE id = ?
                            """,
                                (i, row[0]),
                            )

                        conn.commit()
                        logger.info(
                            f"Rebuilt vector index with {len(contents)} vectors"
                        )

        except Exception as e:
            logger.error(f"Error rebuilding vector index: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                total_embeddings = cursor.fetchone()[0]

                cursor = conn.execute(
                    """
                    SELECT source_type, COUNT(*)
                    FROM embeddings
                    GROUP BY source_type
                """
                )

                source_stats = {}
                for row in cursor:
                    source_stats[row[0] or "unknown"] = row[1]

                return {
                    "total_embeddings": total_embeddings,
                    "vector_index_size": (
                        self.vector_index.ntotal if self.vector_index else 0
                    ),
                    "embedding_dimension": self.dimension,
                    "model_name": self.embedding_model_name,
                    "source_stats": source_stats,
                    "faiss_available": FAISS_AVAILABLE,
                    "embeddings_available": SENTENCE_TRANSFORMERS_AVAILABLE,
                }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

    def _get_content_hash(self, content: str) -> str:
        """Get SHA256 hash of content."""
        import hashlib

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def cleanup_old_embeddings(self, days_old: int = 30) -> int:
        """
        Clean up old embeddings.

        Args:
            days_old: Age threshold in days

        Returns:
            Number of deleted embeddings
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"""
                    DELETE FROM embeddings
                    WHERE created_at < datetime('now', '-{days_old} days')
                """
                )

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old embeddings")
                return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old embeddings: {e}")
            return 0
