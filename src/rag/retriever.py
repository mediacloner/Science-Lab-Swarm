"""RAG retriever for long analysis sessions — wraps DocumentIndexer for agent use."""

import logging

from src.ingestion.indexer import DocumentIndexer

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant document chunks for agent context during analysis."""

    def __init__(self, persist_dir: str = "output/knowledge_base", embedding_model: str = "all-MiniLM-L6-v2"):
        self.indexer = DocumentIndexer(persist_dir=persist_dir, embedding_model=embedding_model)

    def retrieve(self, query: str, collection_name: str = "lab_documents", top_k: int = 10) -> str:
        """Query the knowledge base and return formatted context string."""
        results = self.indexer.query(query, collection_name=collection_name, top_k=top_k)

        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "unknown")
            section = r["metadata"].get("section", "")
            distance = r.get("distance", 0)
            relevance = f"(relevance: {1 - distance:.2f})" if distance else ""
            parts.append(f"[{i}] {source} — {section} {relevance}\n{r['text']}")

        return "\n\n---\n\n".join(parts)

    def retrieve_for_agent(self, agent_analysis: str, topic: str, collection_name: str = "lab_documents", top_k: int = 5) -> str:
        """Retrieve chunks relevant to both the topic and the agent's current analysis."""
        # Combine topic and recent analysis for better retrieval
        query = f"{topic}\n{agent_analysis[:500]}"
        return self.retrieve(query, collection_name=collection_name, top_k=top_k)
