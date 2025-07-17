import chromadb
from sentence_transformers import SentenceTransformer  
from pathlib import Path
from typing import List, Dict, Any
import json
from src.constants import CHUNK_RECORDS_FILE, COLLECTION_NAME


class VectorStore:
    """A thin wrapper around ChromaDB for textual chunks."""

    def __init__(self, persist_directory: str = "vector_store", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", st = None) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.embedder = SentenceTransformer(model_name)
        self.st = st

    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for k, v in meta.items():
            if k == "text": continue
            if v is None: continue
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                try: sanitized[k] = json.dumps(v, ensure_ascii=False)
                except TypeError: sanitized[k] = str(v)
        return sanitized

    def _embed_texts(self, texts: List[str]):
        try:
            embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            dim = self.embedder.get_sentence_embedding_dimension() or 384
            return [[0.0] * dim for _ in texts]

    def add_documents(self, docs: List[Dict[str, Any]]):
        embeddings = self._embed_texts([d["text"] for d in docs])
        sanitized_metas = [self._sanitize_metadata(d) for d in docs]
        self.collection.add(
            ids=[d["id"] for d in docs],
            embeddings=embeddings,
            metadatas=sanitized_metas,
        )

    def ingest_from_jsonl(self, jsonl_path: Path = CHUNK_RECORDS_FILE, batch_size: int = 64):
        buffer: List[Dict[str, Any]] = []
        if not jsonl_path.exists():
            if self.st:
                self.st.error(f"Error: Chunk records file not found at {jsonl_path}. Please run `parse_ingest.py` first.")
            else:
                print(f"Error: Chunk records file not found at {jsonl_path}. Please run `parse_ingest.py` first.")
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                buffer.append(json.loads(line))
                if len(buffer) >= batch_size:
                    self.add_documents(buffer)
                    buffer.clear()
        if buffer:
            self.add_documents(buffer)

    def query(self, text: str, k: int = 5, source_pdf: str | None = None):
        embedding = self._embed_texts([text])[0]
        query_kwargs = dict(
            query_embeddings=[embedding],
            n_results=k,
            include=["metadatas", "distances"],
        )
        if source_pdf:
            query_kwargs["where"] = {"source_pdf": source_pdf}

        res = self.collection.query(**query_kwargs)

        meta_lists = res.get("metadatas", []) or []
        dist_lists = res.get("distances", []) or []

        if not meta_lists:
            return []

        first_meta_list = meta_lists[0]
        first_dist_list = dist_lists[0] if dist_lists else []

        results: List[Dict[str, Any]] = []
        for meta_raw, dist in zip(first_meta_list, first_dist_list):
            if meta_raw is None: continue
            meta_dict = dict(meta_raw)
            meta_dict["distance"] = dist
            results.append(meta_dict)
        return results