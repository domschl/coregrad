"""Decoupled Retriever interface, fast InMemoryBM25Retriever, and Hybrid EmbeddingGemma 2 + SQLite FTS5 index."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "he", "her", "his", "how", "i",
        "in", "into", "is", "it", "its", "of", "on", "or", "she", "that",
        "the", "their", "them", "they", "this", "to", "was", "we", "were",
        "what", "when", "where", "which", "who", "why", "with", "you", "your",
    }
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    start: int
    end: int
    score: float
    text: str

    def to_compact_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "span": [self.start, self.end],
            "text": self.text,
        }


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[SearchHit]: ...

    def read(self, doc_id: str, start: int = 0, length: int = 800) -> dict[str, object]: ...

    def list_docs(self, filter_text: str = "", limit: int = 20) -> list[dict[str, str]]: ...


class Embedder(Protocol):
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingGemma2Embedder:
    """Lazy wrapper for google/embeddinggemma-2 (Apache-2.0, text-only, Matryoshka MRL)."""

    def __init__(
        self,
        model_id: str = "google/embeddinggemma-2",
        mrl_dim: int = 256,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.dim = mrl_dim
        self._device = device
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id,
                truncate_dim=self.dim,
                device=self._device,
            )
        return self._model

    def _encode(self, texts: list[str], prompt_name: str | None = None) -> np.ndarray:
        model = self._load()
        kwargs: dict[str, object] = {"normalize_embeddings": True, "convert_to_numpy": True}
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        try:
            arr = model.encode(texts, **kwargs)  # type: ignore[attr-defined]
        except (KeyError, ValueError, TypeError):
            kwargs.pop("prompt_name", None)
            arr = model.encode(texts, **kwargs)  # type: ignore[attr-defined]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        # Ensure MRL truncation + unit L2 normalization
        if arr.shape[1] > self.dim:
            arr = arr[:, : self.dim]
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prompt_name="document")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prompt_name="query")


def _tokenize_bm25(text: str, drop_stopwords: bool = False) -> list[str]:
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    if not drop_stopwords:
        return toks
    filtered = [t for t in toks if t not in _STOPWORDS and len(t) > 1]
    return filtered or toks


def chunk_markdown_text(
    text: str, chunk_chars: int = 1000, overlap_chars: int = 150
) -> list[tuple[int, int, str]]:
    """Split document into compact ~250-token passages preserving exact character offsets."""
    n = len(text)
    if n == 0:
        return []
    if n <= chunk_chars:
        return [(0, n, text.strip())]

    chunks: list[tuple[int, int, str]] = []
    pos = 0
    while pos < n:
        end = min(n, pos + chunk_chars)
        if end < n:
            # Prefer breaking on paragraph or sentence boundary
            para_break = text.rfind("\n\n", pos + chunk_chars // 2, end)
            if para_break != -1:
                end = para_break
            else:
                sent_break = text.rfind(". ", pos + chunk_chars // 2, end)
                if sent_break != -1:
                    end = sent_break + 1
        snippet = text[pos:end].strip()
        if snippet:
            chunks.append((pos, end, snippet))
        if end >= n:
            break
        pos = max(pos + 1, end - overlap_chars)
    return chunks


def extract_peak_excerpt(text: str, query: str, max_chars: int = 900) -> str:
    """Trim a long external chunk around the highest query-term overlap window."""
    if len(text) <= max_chars:
        return text
    q_terms = set(_tokenize_bm25(query))
    if not q_terms:
        return text[:max_chars]

    step = 150
    best_pos = 0
    best_score = -1
    for start in range(0, max(1, len(text) - max_chars + 1), step):
        window = text[start : start + max_chars]
        w_terms = set(_tokenize_bm25(window))
        score = len(q_terms & w_terms)
        if score > best_score:
            best_score = score
            best_pos = start
    return text[best_pos : best_pos + max_chars].strip()


class InMemoryBM25Retriever:
    """Zero-I/O pure-Python BM25 retriever for synthetic trajectory generation & GRPO."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs: dict[str, Document] = {d.doc_id: d for d in docs}
        self.k1 = k1
        self.b = b
        self._chunks: list[tuple[str, str, int, int, str, Counter[str], int]] = []
        self._df: Counter[str] = Counter()

        for doc in docs:
            for start, end, snippet in chunk_markdown_text(doc.text, chunk_chars=900, overlap_chars=120):
                toks = _tokenize_bm25(f"{doc.title} {snippet}")
                tf = Counter(toks)
                self._chunks.append((doc.doc_id, doc.title, start, end, snippet, tf, len(toks)))
                for term in tf:
                    self._df[term] += 1

        self._n_chunks = max(len(self._chunks), 1)
        self._avgdl = sum(c[6] for c in self._chunks) / self._n_chunks

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        q_terms = _tokenize_bm25(query, drop_stopwords=True)
        if not q_terms or not self._chunks:
            return []

        scored: list[tuple[float, int]] = []
        for idx, (_, _, _, _, _, tf, dl) in enumerate(self._chunks):
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                df = self._df[term]
                idf = math.log(1.0 + (self._n_chunks - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-6))
                score += idf * (f * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scored.append((score, idx))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits: list[SearchHit] = []
        for score, idx in scored[: max(1, min(k, 10))]:
            doc_id, title, start, end, snippet, _, _ = self._chunks[idx]
            hits.append(SearchHit(doc_id=doc_id, title=title, start=start, end=end, score=score, text=snippet))
        return hits

    def read(self, doc_id: str, start: int = 0, length: int = 800) -> dict[str, object]:
        doc = self.docs.get(doc_id)
        if doc is None:
            return {"error": f"Document not found: {doc_id}"}
        lo = max(0, start)
        hi = min(len(doc.text), lo + max(1, min(length, 2000)))
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "span": [lo, hi],
            "total_chars": len(doc.text),
            "text": doc.text[lo:hi],
        }

    def list_docs(self, filter_text: str = "", limit: int = 20) -> list[dict[str, str]]:
        needle = filter_text.strip().lower()
        out: list[dict[str, str]] = []
        for doc in self.docs.values():
            if not needle or needle in doc.title.lower() or needle in doc.doc_id.lower():
                out.append({"doc_id": doc.doc_id, "title": doc.title})
                if len(out) >= limit:
                    break
        return out


class HybridMarkdownIndex:
    """Persistent SQLite FTS5 BM25 + optional EmbeddingGemma 2 dense vector index fused via RRF."""

    def __init__(self, index_dir: Path, embedder: Embedder | None = None) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / "index.sqlite"
        self.vec_path = self.index_dir / "embeddings.npy"
        self.embedder = embedder
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
        self._vectors: np.ndarray | None = None
        if self.vec_path.exists():
            self._vectors = np.load(self.vec_path, mmap_mode="r")

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                title, text, content='chunks', content_rowid='chunk_id',
                tokenize='porter unicode61'
            );
            """
        )
        self._conn.commit()

    def build_from_documents(
        self,
        docs: list[Document],
        chunk_chars: int = 1000,
        overlap_chars: int = 150,
        embed_batch_size: int = 64,
    ) -> dict[str, int]:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM docs")
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM chunks_fts")

        chunk_texts: list[str] = []
        for doc in docs:
            cur.execute(
                "INSERT INTO docs(doc_id, title, source, text) VALUES (?, ?, ?, ?)",
                (doc.doc_id, doc.title, doc.source, doc.text),
            )
            for start, end, snippet in chunk_markdown_text(doc.text, chunk_chars, overlap_chars):
                cur.execute(
                    "INSERT INTO chunks(doc_id, title, start_char, end_char, text) VALUES (?, ?, ?, ?, ?)",
                    (doc.doc_id, doc.title, start, end, snippet),
                )
                rowid = cur.lastrowid
                cur.execute(
                    "INSERT INTO chunks_fts(rowid, title, text) VALUES (?, ?, ?)",
                    (rowid, doc.title, snippet),
                )
                chunk_texts.append(f"{doc.title}\n{snippet}")
        self._conn.commit()

        if self.embedder is not None and chunk_texts:
            batches: list[np.ndarray] = []
            for i in range(0, len(chunk_texts), embed_batch_size):
                batches.append(self.embedder.encode_documents(chunk_texts[i : i + embed_batch_size]))
            mat = np.concatenate(batches, axis=0).astype(np.float32)
            np.save(self.vec_path, mat)
            self._vectors = np.load(self.vec_path, mmap_mode="r")
        elif self.vec_path.exists():
            self.vec_path.unlink()
            self._vectors = None

        return {"docs": len(docs), "chunks": len(chunk_texts), "has_vectors": int(self._vectors is not None)}

    def index_markdown_directory(
        self,
        root_dir: Path,
        chunk_chars: int = 1000,
        overlap_chars: int = 150,
    ) -> dict[str, int]:
        files = sorted(p for p in root_dir.rglob("*.md") if p.is_file())
        docs: list[Document] = []
        for idx, path in enumerate(files, start=1):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            title = path.stem.replace("_", " ").replace("-", " ")
            for line in text.splitlines()[:5]:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            docs.append(
                Document(
                    doc_id=f"d{idx:04d}",
                    title=title,
                    text=text,
                    source=str(path.relative_to(root_dir)),
                )
            )
        return self.build_from_documents(docs, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

    def _bm25_ranks(self, query: str, limit: int = 20) -> list[int]:
        terms = _tokenize_bm25(query, drop_stopwords=True)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms[:16])
        rows = self._conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def _dense_ranks(self, query: str, limit: int = 20) -> list[int]:
        if self.embedder is None or self._vectors is None or len(self._vectors) == 0:
            return []
        q_vec = self.embedder.encode_queries([query])[0]
        scores = self._vectors @ q_vec
        top_n = min(limit, len(scores))
        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        # SQLite autoincrement chunk_id starts at 1 after fresh build
        return [int(i) + 1 for i in top_idx]

    def search(self, query: str, k: int = 3, rrf_k: int = 60) -> list[SearchHit]:
        pool = max(20, k * 5)
        bm25_ids = self._bm25_ranks(query, limit=pool)
        dense_ids = self._dense_ranks(query, limit=pool)

        rrf_scores: dict[int, float] = {}
        for rank, cid in enumerate(bm25_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, cid in enumerate(dense_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

        if not rrf_scores:
            return []

        top_cids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[: max(1, min(k, 10))]
        hits: list[SearchHit] = []
        for cid in top_cids:
            row = self._conn.execute(
                "SELECT doc_id, title, start_char, end_char, text FROM chunks WHERE chunk_id = ?",
                (cid,),
            ).fetchone()
            if row is not None:
                hits.append(
                    SearchHit(
                        doc_id=row[0],
                        title=row[1],
                        start=int(row[2]),
                        end=int(row[3]),
                        score=rrf_scores[cid],
                        text=row[4],
                    )
                )
        return hits

    def read(self, doc_id: str, start: int = 0, length: int = 800) -> dict[str, object]:
        row = self._conn.execute(
            "SELECT title, text FROM docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return {"error": f"Document not found: {doc_id}"}
        title, text = row
        lo = max(0, start)
        hi = min(len(text), lo + max(1, min(length, 2000)))
        return {
            "doc_id": doc_id,
            "title": title,
            "span": [lo, hi],
            "total_chars": len(text),
            "text": text[lo:hi],
        }

    def list_docs(self, filter_text: str = "", limit: int = 20) -> list[dict[str, str]]:
        needle = f"%{filter_text.strip()}%"
        rows = self._conn.execute(
            "SELECT doc_id, title FROM docs WHERE title LIKE ? OR doc_id LIKE ? ORDER BY doc_id LIMIT ?",
            (needle, needle, limit),
        ).fetchall()
        return [{"doc_id": r[0], "title": r[1]} for r in rows]
