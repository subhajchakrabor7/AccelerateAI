"""Semantic memory stub -- no-ops when ChromaDB is unavailable.

The new pipeline does not depend on ChromaDB. This module is kept for
backward compatibility with any legacy code that imports store_document or
query_memory, but all calls are silently no-ops so the pipeline does not
fail if chromadb is missing or misconfigured.
"""

from core.logger import log


def store_document(doc_id: str, text: str, metadata: dict = None) -> None:
    """Store a document in semantic memory (no-op if backend is unavailable)."""
    try:
        import chromadb
        from core.config import BASE_DIR
        chroma_dir = BASE_DIR / ".chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection("idamp_memory")
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
    except Exception as exc:
        log(f"[MEMORY] store_document skipped: {exc}")


def query_memory(query_text: str, n_results: int = 5) -> list:
    """Query semantic memory (returns empty list if backend is unavailable)."""
    try:
        import chromadb
        from core.config import BASE_DIR
        chroma_dir = BASE_DIR / ".chroma"
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection("idamp_memory")
        results = collection.get(limit=n_results)
        return [
            {"id": i, "document": d, "metadata": m}
            for i, d, m in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]
    except Exception as exc:
        log(f"[MEMORY] query_memory skipped: {exc}")
        return []
