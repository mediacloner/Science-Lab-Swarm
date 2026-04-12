"""ChromaDB vector indexer for document chunks."""

import logging
import os
import warnings
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the embedding model (CPU-only)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        # Suppress noisy HF/exllamav2 output during load (progress bars,
        # LOAD REPORT tables, unauthenticated-request warnings).
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        import contextlib
        import io
        from sentence_transformers import SentenceTransformer
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _embedding_model = SentenceTransformer(model_name, device="cpu")
        logger.info(f"Embedding model ready: {model_name}")
    return _embedding_model


class DocumentIndexer:
    """Indexes document chunks into ChromaDB for RAG retrieval."""

    def __init__(self, persist_dir: str = "output/knowledge_base", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._embedding_model_name = embedding_model
        self._model = None

    @property
    def model(self):
        """Lazy-load embedding model on first use (not needed for list/stats)."""
        if self._model is None:
            self._model = get_embedding_model(self._embedding_model_name)
        return self._model

    def index_chunks(self, chunks: list[dict], collection_name: str = "lab_documents") -> int:
        """Add chunks to a ChromaDB collection.

        Returns number of chunks indexed.
        """
        if not chunks:
            return 0

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Prepare batch
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "source_path": c["source_path"],
                "section": c["section"],
                "format": c["metadata"]["format"],
                "chunk_index": c["metadata"]["chunk_index"],
            }
            for c in chunks
        ]

        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        # Upsert to collection
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Indexed {len(chunks)} chunks into collection '{collection_name}'")
        return len(chunks)

    def query(self, query_text: str, collection_name: str = "lab_documents", top_k: int = 10) -> list[dict]:
        """Query the vector store for relevant chunks."""
        collection = self.client.get_or_create_collection(name=collection_name)

        query_embedding = self.model.encode([query_text]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return retrieved

    def list_collections(self) -> list[str]:
        """List all available collections."""
        return [c.name for c in self.client.list_collections()]

    def collection_stats(self, collection_name: str) -> dict:
        """Get stats for a collection."""
        collection = self.client.get_or_create_collection(name=collection_name)
        return {
            "name": collection_name,
            "count": collection.count(),
        }
