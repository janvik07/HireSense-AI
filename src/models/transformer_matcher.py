"""Semantic matching with Sentence Transformers (all-MiniLM-L6-v2).

IMPORTANT: this uses the *pretrained* model as-is. It is NOT fine-tuned here.
Similarity is the cosine between the job-description embedding and resume
chunk embeddings; it is a relevance signal, not a hiring probability.

Long resumes are split into overlapping word chunks (the model only reads
~256 word-pieces), every chunk is embedded, and the resume score is the mean of
its top-k most similar chunks. The best chunks double as supporting evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.preprocessing import split_into_chunks

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache so the model is loaded once per process, not per request.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class ModelLoadError(RuntimeError):
    """Raised when the transformer model cannot be loaded (e.g. offline on first run)."""


@dataclass
class SemanticScore:
    """Semantic similarity result for a single resume."""

    similarity: float  # 0..1 (negative cosines are clipped to 0)
    best_chunks: list[tuple[str, float]] = field(default_factory=list)  # (text, cosine)
    num_chunks: int = 0


def resolve_device(requested: str = "auto") -> str:
    """'auto' -> 'cuda' if available else 'cpu'. Explicit values are passed through."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def load_sentence_model(model_name: str = DEFAULT_MODEL_NAME, device: str = "auto") -> Any:
    """Load (and cache) a SentenceTransformer. The first call downloads ~90 MB to the
    Hugging Face cache (~/.cache/huggingface); later calls are offline-capable."""
    resolved = resolve_device(device)
    key = (model_name, resolved)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=resolved)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load '{model_name}' ({type(exc).__name__}: {exc}). "
            "The first run needs internet access to download the model; afterwards it is cached."
        ) from exc
    _MODEL_CACHE[key] = model
    return model


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TransformerMatcher:
    """Scores resumes against a job description using sentence embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        model: Any | None = None,
        chunk_words: int = 150,
        chunk_overlap: int = 30,
        top_k_chunks: int = 3,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = model  # allows injecting a fake model in tests
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
        self.top_k_chunks = top_k_chunks
        self.batch_size = batch_size

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = load_sentence_model(self.model_name, self.device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def _chunks(self, text: str) -> list[str]:
        return split_into_chunks(text, self.chunk_words, self.chunk_overlap) or [text]

    def score(self, job_description: str, resumes: list[str]) -> list[SemanticScore]:
        """Score every resume against the job description (one batched encode call)."""
        if not job_description.strip():
            raise ValueError("Job description is empty.")

        # Job description -> mean of its chunk embeddings (handles long JDs).
        jd_embeddings = self.embed(self._chunks(job_description))
        jd_vector = _l2_normalize(jd_embeddings.mean(axis=0, keepdims=True))[0]

        # Flatten all resume chunks into a single batch for speed.
        all_chunks: list[str] = []
        owners: list[int] = []
        for resume_index, resume in enumerate(resumes):
            for chunk in self._chunks(resume):
                all_chunks.append(chunk)
                owners.append(resume_index)

        results: list[SemanticScore] = [SemanticScore(similarity=0.0) for _ in resumes]
        if not all_chunks:
            return results

        chunk_embeddings = self.embed(all_chunks)
        similarities = chunk_embeddings @ jd_vector  # cosine, since both are unit length
        owners_array = np.asarray(owners)

        for resume_index in range(len(resumes)):
            positions = np.where(owners_array == resume_index)[0]
            if positions.size == 0:
                continue
            sims = similarities[positions]
            order = np.argsort(-sims)
            top = order[: self.top_k_chunks]
            score = float(np.clip(sims[top].mean(), 0.0, 1.0))
            best = [(all_chunks[positions[i]], float(sims[i])) for i in top]
            results[resume_index] = SemanticScore(
                similarity=score, best_chunks=best, num_chunks=int(positions.size)
            )
        return results
