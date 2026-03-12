# src/chroma_manager.py

"""
ChromaDB manager for Medical RAG experiments
"""

import logging
from pathlib import Path
from typing import Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


SUPPORTED_MODELS = {
    "pubmedbert": "pritamdeka/S-PubMedBert-MS-MARCO",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-base-v2",
    "specter2": "allenai/specter2_base",
}


class ChromaManager:

    def __init__(
        self,
        base_directory: Path,
        chunk_strategy: str,
        embedding_model: str = "pubmedbert",
        collection_name: str = "medical_rag",
    ):
    
        if embedding_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{embedding_model}'. "
                f"Choose from {list(SUPPORTED_MODELS.keys())}"
            )
    
        self.chunk_strategy = chunk_strategy
        self.embedding_model_key = embedding_model
        self.embedding_model_name = SUPPORTED_MODELS[embedding_model]
    
        # Build experiment path
        self.persist_directory = base_directory / chunk_strategy / embedding_model
        self.persist_directory.mkdir(parents=True, exist_ok=True)
    
        print(f"\nLoading embedding model: {self.embedding_model_name}")
    
        self.embedder = SentenceTransformer(self.embedding_model_name)
    
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(allow_reset=True)
        )

        self.collection_name = collection_name
    
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
        print(
            f"Vector DB ready → {self.persist_directory}"
        )

    # ---------------------------------------------------------
    # Bulk insert chunks with progress
    # ---------------------------------------------------------

    def add_chunks(self, chunk_df, batch_size: int = 256):

        docs = []
        metas = []
        ids = []

        total_rows = len(chunk_df)

        for _, row in tqdm(
            chunk_df.iterrows(),
            total=total_rows,
            desc="Preparing chunks"
        ):

            text = row["chunk_text"]

            metadata = {
                "pmid": row["pmid"],
                "section": row["section"],
                "topic_id": row.get("topic_id"),
                "category": row.get("category"),
            }

            docs.append(text)
            metas.append(metadata)
            ids.append(row["chunk_id"])

            if len(docs) >= batch_size:

                self._insert_batch(docs, metas, ids)

                docs, metas, ids = [], [], []

        if docs:
            self._insert_batch(docs, metas, ids)

        print("\nFinished inserting all chunks.")

    # ---------------------------------------------------------
    # Batch insert with embedding progress
    # ---------------------------------------------------------

    def _insert_batch(self, docs, metas, ids):

        embeddings = self.embedder.encode(
            docs,
            show_progress_bar=True,   # embedding progress
            convert_to_numpy=True
        )

        self.collection.add(
            documents=docs,
            embeddings=embeddings.tolist(),
            metadatas=metas,
            ids=ids
        )

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    def query(self, query_text: str, n_results: int = 5):

        query_embedding = self.embedder.encode([query_text])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )

        return results

    # ---------------------------------------------------------
    # Stats
    # ---------------------------------------------------------

    def stats(self):

        return {
            "collection": self.collection_name,
            "embedding_model": self.embedding_model_key,
            "count": self.collection.count(),
        }

    # ---------------------------------------------------------
    # Reset collection
    # ---------------------------------------------------------

    def reset(self):

        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        print("Collection reset.")