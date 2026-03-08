"""
Medical Embedding Manager supporting multiple HuggingFace models.
Optimized for Medical RAG experiments.
"""

import logging
import os
from typing import List, Optional
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------------------------------
# Cache configuration
# -------------------------------------------------

CACHE_DIR = Path(os.getenv("TRANSFORMERS_CACHE", ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Supported embedding models
# -------------------------------------------------

SUPPORTED_MODELS = {
    "pubmedbert": "pritamdeka/S-PubMedBert-MS-MARCO",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-base-v2",
    "specter2": "allenai/specter2_base",
}


class MedicalEmbeddingManager:
    """
    Embedding manager for Medical RAG pipelines.
    Supports multiple HuggingFace sentence-transformer models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
        batch_size: Optional[int] = None,
        max_chars: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):

        # -------------------------------------------------
        # Load configuration from .env
        # -------------------------------------------------

        self.model_key = model_name or os.getenv("EMBEDDING_MODEL", "pubmedbert")

        self.batch_size = batch_size or int(
            os.getenv("EMBEDDING_BATCH_SIZE", 32)
        )

        self.max_chars = max_chars or int(
            os.getenv("EMBEDDING_MAX_CHARS", 8000)
        )

        if normalize is None:
            normalize = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"

        self.normalize = normalize

        # -------------------------------------------------
        # Logger
        # -------------------------------------------------

        self.logger = logger or logging.getLogger(__name__)

        # -------------------------------------------------
        # Validate model
        # -------------------------------------------------

        if self.model_key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported embedding model '{self.model_key}'. "
                f"Choose from: {list(SUPPORTED_MODELS.keys())}"
            )

        self.embedding_model = SUPPORTED_MODELS[self.model_key]

        # -------------------------------------------------
        # Device selection
        # -------------------------------------------------

        device_env = os.getenv("EMBEDDING_DEVICE", "auto")

        if device is None:
            if device_env == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = device_env

        self.device = device

        # -------------------------------------------------
        # Load embedding model
        # -------------------------------------------------

        self.logger.info(
            "Loading embedding model '%s' (%s) on %s",
            self.model_key,
            self.embedding_model,
            self.device,
        )

        self.model = SentenceTransformer(
            self.embedding_model,
            device=self.device,
            cache_folder=str(CACHE_DIR / "sentence_transformers"),
        )

        self.logger.info(
            "Embedding manager ready (dim=%d)",
            self.model.get_sentence_embedding_dimension(),
        )

    # -------------------------------------------------
    # Text preprocessing
    # -------------------------------------------------

    def _truncate(self, text: str) -> str:
        """
        Clean and truncate text input.
        """
        text = text.strip()

        if len(text) > self.max_chars:
            return text[: self.max_chars]

        return text

    def _prepare_query(self, text: str) -> str:
        """
        Prepare query for specific embedding models.
        """

        if self.model_key in {"bge", "e5"}:
            return f"query: {text}"

        return text

    def _prepare_document(self, text: str) -> str:
        """
        Prepare document text for embedding models.
        """

        if self.model_key in {"bge", "e5"}:
            return f"passage: {text}"

        return text

    # -------------------------------------------------
    # Query embedding
    # -------------------------------------------------

    def embed_query(self, query: str) -> List[float]:

        if not query or not query.strip():
            return []

        try:

            query = self._prepare_query(query)
            query = self._truncate(query)

            with torch.inference_mode():
                embedding = self.model.encode(
                    query,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True,
                )

            return embedding.tolist()

        except Exception:
            self.logger.exception("Query embedding failed")
            return []

    # -------------------------------------------------
    # Document embedding
    # -------------------------------------------------

    def embed_document(self, text: str) -> List[float]:

        if not text or not text.strip():
            return []

        try:

            text = self._prepare_document(text)
            text = self._truncate(text)

            with torch.inference_mode():
                embedding = self.model.encode(
                    text,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True,
                )

            return embedding.tolist()

        except Exception:
            self.logger.exception("Document embedding failed")
            return []

    # -------------------------------------------------
    # Batch embedding
    # -------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        if not texts:
            return []

        try:

            texts = [self._prepare_document(t) for t in texts]
            texts = [self._truncate(t) for t in texts]

            with torch.inference_mode():
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 32,
                )

            return embeddings.tolist()

        except Exception:
            self.logger.exception("Batch embedding failed")
            return []

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    @property
    def embedding_dimension(self) -> int:
        """
        Return embedding vector dimension.
        """
        return self.model.get_sentence_embedding_dimension()