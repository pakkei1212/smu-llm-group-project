# src/chroma_manager.py
"""
ChromaDB manager module for storing and retrieving vector embeddings.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Add project root to path for config import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
VERBOSE_LEVEL_MAP = {
    0: logging.CRITICAL + 1,  # effectively silence everything
    1: logging.INFO,
    2: logging.DEBUG,
}

logger = logging.getLogger(__name__)

class ChromaManager:
    """
    Manager for handling ChromaDB operations for vector storage and retrieval.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "documents",
        base_url: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Initialize the ChromaDB manager.

        Args:
            persist_directory: Directory to persist ChromaDB (defaults to VECTOR_DB_PATH)
            embedding_model: Ollama model to use for embeddings
            collection_name: Name of the collection to use
            base_url: Base URL of the Ollama API (defaults to env var OLLAMA_HOST or localhost)
        """
        # ------------------------------
        # Configure logging FIRST
        # ------------------------------
        self.verbose = verbose
        self._configure_logger(verbose)
    
        # ------------------------------
        # Resolve paths and configuration
        # ------------------------------
        self.persist_directory = persist_directory or Path("/app/RAG/chroma_db")
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Prefer explicit arg → env var → default localhost
        host = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.api_url = f"{host.rstrip('/')}/api/embeddings"

        # ------------------------------
        # Ensure persistence directory
        # ------------------------------
        os.makedirs(self.persist_directory, exist_ok=True)
        try:
            os.chmod(self.persist_directory, 0o777)
            logger.debug(f"Persist directory: {self.persist_directory}")
        except Exception:
            pass  # Safe fallback if running on read-only FS

        # ------------------------------
        # Initialize ChromaDB client
        # ------------------------------
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(allow_reset=True)
            )         
            logger.debug("Chroma PersistentClient initialized successfully")
        except Exception as e:
            logger.warning(f"PersistentClient init failed ({e}); resetting directory.")
            import shutil
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(allow_reset=True)
            )

        # ------------------------------
        # Initialize embedding function
        # ------------------------------
        self.embedding_function = OllamaEmbeddingFunction(
            model_name=self.embedding_model,
            url=self.api_url
        )

        # ------------------------------
        # Get or create collection
        # ------------------------------
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to load collection '{self.collection_name}' ({e}); creating new one.")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

        logger.info(
            f"ChromaManager initialized with collection: {self.collection_name}, "
            f"embedding model: {self.embedding_model}, api_url: {self.api_url}"
        )

    def _configure_logger(self, verbose: int):
        level = VERBOSE_LEVEL_MAP.get(verbose, logging.INFO)
    
        logger.setLevel(level)
    
        # Prevent duplicate handlers (important in Jupyter)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(levelname)s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            handler.setLevel(level)
            logger.addHandler(handler)
    
        logger.debug(
            f"Logger configured | verbose={verbose} | level={logging.getLevelName(level)}"
        )

    def add_text(self, 
                text: str, 
                metadata: Dict[str, Any], 
                id: str) -> bool:
        """
        Add text content to the collection
        
        Args:
            text: Text content to add
            metadata: Additional metadata for the document
            id: Unique identifier for the document
            
        Returns:
            Success status
        """
        try:
            logger.debug(
                f"add_text | id={id} | text_len={len(text)} | metadata_keys={list(metadata.keys())}"
            )

            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[id]
            )
            logger.info(f"Added text with ID: {id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add text with ID {id}: {e}")
            return False
    
    def add_texts(self, 
                 texts: List[str], 
                 metadatas: List[Dict[str, Any]], 
                 ids: List[str]) -> bool:
        """
        Add multiple text contents to the collection
        
        Args:
            texts: List of text contents to add
            metadatas: List of metadata dictionaries
            ids: List of unique identifiers
            
        Returns:
            Success status
        """
        if not (len(texts) == len(metadatas) == len(ids)):
            logger.error(f"Lengths don't match: texts={len(texts)}, metadatas={len(metadatas)}, ids={len(ids)}")
            return False
        
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} texts to collection")
            return True
        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            return False
    
    def add_with_embedding(self,
                         text: str,
                         embedding: List[float],
                         metadata: Dict[str, Any],
                         id: str) -> bool:
        """
        Add text content with pre-computed embedding
        
        Args:
            text: Text content to add
            embedding: Pre-computed embedding vector
            metadata: Additional metadata for the document
            id: Unique identifier for the document
            
        Returns:
            Success status
        """
        try:
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[id]
            )
            logger.info(f"Added text with custom embedding, ID: {id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add text with custom embedding, ID {id}: {e}")
            return False
    
    def add_with_embeddings(self,
                          texts: List[str],
                          embeddings: List[List[float]],
                          metadatas: List[Dict[str, Any]],
                          ids: List[str]) -> bool:
        """
        Add multiple text contents with pre-computed embeddings
        
        Args:
            texts: List of text contents to add
            embeddings: List of pre-computed embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique identifiers
            
        Returns:
            Success status
        """
        if not (len(texts) == len(embeddings) == len(metadatas) == len(ids)):
            logger.error(f"Lengths don't match: texts={len(texts)}, embeddings={len(embeddings)}, "
                       f"metadatas={len(metadatas)}, ids={len(ids)}")
            return False
        
        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} texts with custom embeddings")
            return True
        except Exception as e:
            logger.error(f"Failed to add texts with custom embeddings: {e}")
            return False
    
    def query(self,
             query_text: str,
             n_results: int = 3,
             where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the collection using text
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            where: Optional filter criteria
            
        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            logger.info(f"Query returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Failed to query: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def query_with_embedding(self,
                           query_embedding: List[float],
                           n_results: int = 3,
                           where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the collection using a pre-computed embedding
        
        Args:
            query_embedding: Pre-computed embedding vector
            n_results: Number of results to return
            where: Optional filter criteria
            
        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            logger.info(f"Embedding query returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Failed to query with embedding: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample item to determine embedding dimension
            sample = self.collection.get(limit=1)
            embedding_dim = "unknown"
            if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                embedding_dim = len(sample["embeddings"][0])
            
            stats = {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "embedding_dimension": embedding_dim,
                "persist_directory": str(self.persist_directory)
            }
            
            logger.info(f"Collection stats: {count} items")
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def reset_collection(self):
        """
        Remove ALL embeddings by deleting and recreating the collection.
        Safe for all Chroma versions.
        """
        name = self.collection_name

        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection '{name}'.")
        except Exception as e:
            logger.warning(f"Collection '{name}' did not exist or could not be deleted ({e}).")
    
        # Recreate empty collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name,
            embedding_function=self.embedding_function
        )
    
        logger.info(f"Recreated empty collection '{name}'.")
    
        return True