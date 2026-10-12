import numpy as np
import pytest

from src.demo_data import DEMO_JOB_DESCRIPTION, DEMO_RESUMES, demo_relevance
from src.evaluation import ndcg_at_k, precision_at_k, spearman_correlation
from src.models.transformer_matcher import TransformerMatcher
from src.parser import ParsedDocument
from src.ranking import (
    METHOD_TFIDF,
    METHOD_TRANSFORMER,
    RankingConfig,
    combine_scores,
    rank_resumes,
    results_to_dataframe,
    tfidf_similarities,
)

JD = "We need a Python developer with PyTorch, scikit-learn and SQL experience for NLP work."
GOOD = "Python engineer. Built NLP models with PyTorch and scikit-learn, queried data with SQL."
BAD = "Graphic designer skilled in Photoshop, illustration and branding for print campaigns."


def docs(*texts):
    return [ParsedDocument(filename=f"r{i}.txt", text=t) for i, t in enumerate(texts)]


def test_config_normalises_weights():
    assert RankingConfig(3, 1).normalized() == (0.75, 0.25)
    with pytest.raises(ValueError):
        RankingConfig(0, 0).normalized()
    with pytest.raises(ValueError):
        RankingConfig(-1, 1).normalized()


def test_combine_scores_formula_and_no_required_skills():
    final, w_sim, w_skill = combine_scores(0.5, 1.0, RankingConfig(0.6, 0.4))
    assert final == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)
    assert (w_sim, w_skill) == (pytest.approx(0.6), pytest.approx(0.4))
    final, w_sim, w_skill = combine_scores(0.5, None, RankingConfig())
    assert final == 0.5 and w_skill == 0.0


def test_tfidf_ranks_relevant_resume_higher():
    sims, terms = tfidf_similarities(JD, [GOOD, BAD])
    assert sims[0] > sims[1]
    assert terms[0]  # has overlapping terms as evidence
    assert all(0.0 <= s <= 1.0 for s in sims)


def test_rank_resumes_tfidf_end_to_end():
    results = rank_resumes(JD, docs(BAD, GOOD))
    assert [r.rank for r in results] == [1, 2]
    assert results[0].filename == "r1.txt"
    top = results[0]
    assert "python" in top.matched_skills and "pytorch" in top.matched_skills
    assert top.skill_coverage == pytest.approx(1.0)
    assert results[1].missing_skills  # designer lacks the required skills
    assert top.skill_evidence["python"]
    assert top.method == METHOD_TFIDF and top.semantic_similarity is None


def test_required_skills_override():
    results = rank_resumes(JD, docs(GOOD), required_skills=["docker"])
    assert results[0].missing_skills == ["docker"]
    assert results[0].skill_coverage == 0.0


def test_no_required_skills_uses_similarity_only():
    results = rank_resumes(JD, docs(GOOD), required_skills=[])
    assert results[0].skill_coverage is None
    assert results[0].final_score == pytest.approx(results[0].similarity)


def test_failed_documents_are_skipped_and_ids_stay_stable():
    documents = [ParsedDocument(filename="bad.pdf", error="corrupt"), *docs(GOOD)]
    results = rank_resumes(JD, documents)
    assert len(results) == 1
    assert results[0].candidate_id == "Candidate 2"
    assert rank_resumes(JD, [ParsedDocument(filename="bad.pdf", error="x")]) == []


def test_empty_job_description_raises():
    with pytest.raises(ValueError):
        rank_resumes("   ", docs(GOOD))


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        rank_resumes(JD, docs(GOOD), method="magic")


def test_ranking_is_deterministic():
    first = [r.filename for r in rank_resumes(JD, docs(GOOD, BAD, GOOD))]
    second = [r.filename for r in rank_resumes(JD, docs(GOOD, BAD, GOOD))]
    assert first == second


def test_dataframe_export_and_anonymisation():
    results = rank_resumes(JD, docs(GOOD, BAD))
    df = results_to_dataframe(results)
    assert list(df["rank"]) == [1, 2]
    assert {"final_score", "skill_coverage", "matched_skills", "missing_skills"} <= set(df.columns)
    anon = results_to_dataframe(results, anonymize=True)
    assert all(str(v).startswith("Candidate ") for v in anon["candidate"])
    assert "r0.txt" not in anon.to_csv()


def test_scores_do_not_depend_on_filename():
    a = rank_resumes(JD, [ParsedDocument(filename="alice.txt", text=GOOD)])
    b = rank_resumes(JD, [ParsedDocument(filename="zed.txt", text=GOOD)])
    assert a[0].final_score == pytest.approx(b[0].final_score)


# ---- transformer path with a fake encoder (no model download) ----
class FakeModel:
    """Bag-of-keywords 'embedding' so tests are fast and deterministic."""

    KEYWORDS = ["python", "pytorch", "sql", "nlp", "design", "photoshop"]

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([1.0 if k in lowered else 0.0 for k in self.KEYWORDS] + [0.1])
        return np.array(rows, dtype=np.float32)


def test_transformer_matcher_with_fake_model():
    matcher = TransformerMatcher(model=FakeModel())
    scores = matcher.score(JD, [GOOD, BAD])
    assert scores[0].similarity > scores[1].similarity
    assert 0.0 <= scores[1].similarity <= 1.0
    assert scores[0].best_chunks and scores[0].num_chunks >= 1


def test_transformer_handles_long_documents_by_chunking():
    long_resume = BAD + " filler" * 600 + " " + GOOD  # relevant text is far from the start
    matcher = TransformerMatcher(model=FakeModel(), chunk_words=100, chunk_overlap=20)
    score = matcher.score(JD, [long_resume])[0]
    assert score.num_chunks > 3
    assert any("pytorch" in chunk.lower() for chunk, _ in score.best_chunks)


def test_rank_resumes_with_transformer_method():
    matcher = TransformerMatcher(model=FakeModel())
    results = rank_resumes(JD, docs(BAD, GOOD), method=METHOD_TRANSFORMER, matcher=matcher)
    assert results[0].filename == "r1.txt"
    assert results[0].semantic_similarity is not None
    assert results[0].tfidf_similarity >= 0.0  # baseline kept for comparison


# ---- evaluation helpers ----
def test_precision_and_ndcg():
    relevance = {"a": 2, "b": 0, "c": 1}
    assert precision_at_k(["a", "c", "b"], relevance, 2) == 1.0
    assert precision_at_k(["b", "a", "c"], relevance, 2) == 0.5
    assert ndcg_at_k(["a", "c", "b"], relevance, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "c", "a"], relevance, 3) < 1.0
    with pytest.raises(ValueError):
        precision_at_k(["a"], relevance, 0)


def test_spearman():
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_correlation([1], [1]) is None


def test_demo_data_is_consistent():
    assert set(demo_relevance()) == set(DEMO_RESUMES)
    results = rank_resumes(DEMO_JOB_DESCRIPTION, [ParsedDocument(filename=n, text=t) for n, (t, _) in DEMO_RESUMES.items()])
    assert len(results) == len(DEMO_RESUMES)
