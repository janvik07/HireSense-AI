"""Ranking: TF-IDF baseline, optional transformer similarity, skill coverage and a
transparent weighted combination.

final_score = w_sim * similarity + w_skill * required_skill_coverage      (weights sum to 1)

All of these numbers are *relevance indicators*. They are NOT probabilities that
a candidate will be hired or will succeed in the role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.parser import ParsedDocument
from src.preprocessing import clean_text
from src.skill_extractor import compare_skills, extract_skills, find_skill_evidence, skill_coverage

if TYPE_CHECKING:  # avoid importing torch / sentence-transformers for the TF-IDF path
    from src.models.transformer_matcher import TransformerMatcher

METHOD_TFIDF = "tfidf"
METHOD_TRANSFORMER = "transformer"
# Keeps tokens such as c++, c#, node.js, scikit-learn parts (scikit, learn) intact.
TFIDF_TOKEN_PATTERN = r"[a-z0-9][a-z0-9+#]*(?:\.[a-z0-9]+)*"


@dataclass(frozen=True)
class RankingConfig:
    """Weights of the combined score. They are normalised so they always sum to 1."""

    similarity_weight: float = 0.6
    skill_weight: float = 0.4

    def normalized(self) -> tuple[float, float]:
        if self.similarity_weight < 0 or self.skill_weight < 0:
            raise ValueError("Weights must be non-negative.")
        total = self.similarity_weight + self.skill_weight
        if total <= 0:
            raise ValueError("At least one weight must be greater than zero.")
        return self.similarity_weight / total, self.skill_weight / total


@dataclass
class CandidateResult:
    """Everything the UI needs to explain one candidate's score."""

    candidate_id: str
    filename: str
    method: str
    final_score: float
    similarity: float  # similarity used in the final score (TF-IDF or semantic)
    tfidf_similarity: float
    semantic_similarity: float | None
    skill_coverage: float | None
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    skill_evidence: dict[str, list[str]] = field(default_factory=dict)
    top_terms: list[tuple[str, float]] = field(default_factory=list)  # TF-IDF overlap terms
    best_chunks: list[tuple[str, float]] = field(default_factory=list)  # semantic evidence
    rank: int = 0
    cleaned_text: str = ""
    similarity_weight_used: float = 0.0
    skill_weight_used: float = 0.0


def tfidf_similarities(
    job_description: str, resumes: list[str], top_n_terms: int = 8
) -> tuple[np.ndarray, list[list[tuple[str, float]]]]:
    """Cosine similarity of each resume to the JD in TF-IDF space.

    Note: IDF is fitted on the JD + the uploaded resumes, so scores depend on the
    batch and are only meaningful for comparing candidates within one run.

    Returns (similarities, top overlapping terms per resume).
    """
    if not resumes:
        return np.zeros(0), []
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=TFIDF_TOKEN_PATTERN,
    )
    try:
        matrix = vectorizer.fit_transform([job_description, *resumes])
    except ValueError:  # empty vocabulary (e.g. only stop-words)
        return np.zeros(len(resumes)), [[] for _ in resumes]

    jd_vector = matrix[0]
    resume_matrix = matrix[1:]
    sims = cosine_similarity(jd_vector, resume_matrix).ravel()

    vocabulary = np.array(vectorizer.get_feature_names_out())
    jd_dense = jd_vector.toarray().ravel()
    terms: list[list[tuple[str, float]]] = []
    for i in range(resume_matrix.shape[0]):
        contribution = jd_dense * resume_matrix[i].toarray().ravel()
        top_idx = np.argsort(-contribution)[:top_n_terms]
        terms.append([(str(vocabulary[j]), float(contribution[j])) for j in top_idx if contribution[j] > 0])
    return sims, terms


def combine_scores(
    similarity: float, coverage: float | None, config: RankingConfig
) -> tuple[float, float, float]:
    """Weighted score. If no required skills exist, similarity alone is used.

    Returns (final_score, similarity_weight_used, skill_weight_used).
    """
    w_sim, w_skill = config.normalized()
    if coverage is None:
        return float(similarity), 1.0, 0.0
    return float(w_sim * similarity + w_skill * coverage), w_sim, w_skill


def rank_resumes(
    job_description: str,
    documents: list[ParsedDocument],
    method: str = METHOD_TFIDF,
    config: RankingConfig | None = None,
    required_skills: list[str] | None = None,
    matcher: "TransformerMatcher | None" = None,
) -> list[CandidateResult]:
    """Rank successfully parsed documents against a job description.

    Args:
        job_description: raw JD text.
        documents: parsed resumes (documents with errors are ignored).
        method: ``"tfidf"`` or ``"transformer"``.
        config: weights of the combined score.
        required_skills: canonical skills to require; defaults to the skills
            detected in the JD.
        matcher: optional pre-built TransformerMatcher (so the UI can cache it).
    """
    if method not in (METHOD_TFIDF, METHOD_TRANSFORMER):
        raise ValueError(f"Unknown method '{method}'.")
    config = config or RankingConfig()
    jd_clean = clean_text(job_description)
    if not jd_clean:
        raise ValueError("Job description is empty after cleaning.")

    # candidate_id is the position in the *input* list, so anonymised labels are stable.
    valid = [(i, d) for i, d in enumerate(documents) if d.ok]
    if not valid:
        return []
    resumes_clean = [clean_text(d.text) for _, d in valid]

    required = sorted(set(required_skills)) if required_skills is not None else extract_skills(jd_clean)

    tfidf_sims, tfidf_terms = tfidf_similarities(jd_clean, resumes_clean)

    semantic_scores = None
    if method == METHOD_TRANSFORMER:
        if matcher is None:
            from src.models.transformer_matcher import TransformerMatcher

            matcher = TransformerMatcher()
        semantic_scores = matcher.score(jd_clean, resumes_clean)

    results: list[CandidateResult] = []
    for k, (original_index, doc) in enumerate(valid):
        text = resumes_clean[k]
        candidate_skills = extract_skills(text)
        matched, missing, extra = compare_skills(required, candidate_skills)
        coverage = skill_coverage(required, candidate_skills)
        evidence = {skill: find_skill_evidence(text, skill) for skill in matched}

        if semantic_scores is not None:
            sem = semantic_scores[k]
            similarity, semantic_value, chunks = sem.similarity, sem.similarity, sem.best_chunks
        else:
            similarity, semantic_value, chunks = float(tfidf_sims[k]), None, []

        final, w_sim, w_skill = combine_scores(similarity, coverage, config)
        results.append(
            CandidateResult(
                candidate_id=f"Candidate {original_index + 1}",
                filename=doc.filename,
                method=method,
                final_score=final,
                similarity=similarity,
                tfidf_similarity=float(tfidf_sims[k]),
                semantic_similarity=semantic_value,
                skill_coverage=coverage,
                matched_skills=matched,
                missing_skills=missing,
                extra_skills=extra,
                skill_evidence=evidence,
                top_terms=tfidf_terms[k],
                best_chunks=chunks,
                cleaned_text=text,
                similarity_weight_used=w_sim,
                skill_weight_used=w_skill,
            )
        )

    # Deterministic order: score desc, then filename.
    results.sort(key=lambda r: (-r.final_score, r.filename))
    for position, result in enumerate(results, start=1):
        result.rank = position
    return results


def results_to_dataframe(results: list[CandidateResult], anonymize: bool = False) -> pd.DataFrame:
    """Flatten results into a table suitable for display and CSV export."""
    rows = []
    for r in results:
        rows.append(
            {
                "rank": r.rank,
                "candidate": r.candidate_id if anonymize else r.filename,
                "final_score": round(r.final_score, 4),
                "similarity_method": "TF-IDF cosine" if r.method == METHOD_TFIDF else "Transformer cosine",
                "similarity": round(r.similarity, 4),
                "tfidf_similarity": round(r.tfidf_similarity, 4),
                "semantic_similarity": None if r.semantic_similarity is None else round(r.semantic_similarity, 4),
                "skill_coverage": None if r.skill_coverage is None else round(r.skill_coverage, 4),
                "similarity_weight": round(r.similarity_weight_used, 3),
                "skill_weight": round(r.skill_weight_used, 3),
                "matched_skills": ", ".join(r.matched_skills),
                "missing_skills": ", ".join(r.missing_skills),
            }
        )
    return pd.DataFrame(rows)
