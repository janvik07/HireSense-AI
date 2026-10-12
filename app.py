"""HireSense AI - Streamlit dashboard.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.demo_data import DEMO_JOB_DESCRIPTION, DEMO_RESUMES
from src.parser import MAX_FILE_BYTES, ParsedDocument, parse_document
from src.preprocessing import clean_text
from src.ranking import (
    METHOD_TFIDF,
    METHOD_TRANSFORMER,
    CandidateResult,
    RankingConfig,
    rank_resumes,
    results_to_dataframe,
)
from src.skill_extractor import CATEGORY_BY_SKILL, SKILL_TAXONOMY, extract_skills

BILSTM_CHECKPOINT = Path("models_saved/bilstm_resume_category.pt")
METHOD_LABELS = {
    "TF-IDF baseline (scikit-learn)": METHOD_TFIDF,
    "Transformer semantic matching (all-MiniLM-L6-v2)": METHOD_TRANSFORMER,
}
SCORE_DISCLAIMER = (
    "Scores are **relevance indicators** (text similarity + required-skill coverage). "
    "They are **not** hiring probabilities and must not be the only basis for a hiring decision. "
    "A human should review every candidate."
)

st.set_page_config(page_title="HireSense AI", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    .hs-sub {color: #6b7280; margin-top: -0.6rem; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading transformer model (first run downloads ~90 MB)...")
def get_matcher(device: str):
    from src.models.transformer_matcher import TransformerMatcher

    matcher = TransformerMatcher(device=device)
    _ = matcher.model  # force load now so errors surface here
    return matcher


@st.cache_resource(show_spinner=False)
def get_bilstm_predictor(path: str, mtime: float):
    """Load the BiLSTM checkpoint (``mtime`` busts the cache when the file changes)."""
    from src.models.bilstm import BiLSTMPredictor

    return BiLSTMPredictor(path)


def load_bilstm_if_valid():
    """Return a predictor only for a real (non-demo) checkpoint with a label mapping."""
    if not BILSTM_CHECKPOINT.is_file():
        return None, "No trained BiLSTM checkpoint found."
    try:
        predictor = get_bilstm_predictor(str(BILSTM_CHECKPOINT), BILSTM_CHECKPOINT.stat().st_mtime)
    except Exception as exc:
        return None, f"BiLSTM checkpoint could not be loaded ({type(exc).__name__}: {exc})."
    if predictor.is_demo:
        return None, "The checkpoint was trained on synthetic demo data, so its predictions are hidden."
    if not predictor.label_names:
        return None, "The checkpoint has no label mapping, so its predictions are hidden."
    return predictor, ""


# ---------------------------------------------------------------------------
# Header + privacy
# ---------------------------------------------------------------------------
st.title("🎯 HireSense AI")
st.markdown('<p class="hs-sub">Explainable resume screening and job matching</p>', unsafe_allow_html=True)
st.info(SCORE_DISCLAIMER)

with st.expander("🔒 Privacy notice", expanded=False):
    st.markdown(
        """
- Uploaded resumes are processed **in memory on the machine running this app**; this code does not save them to disk or send them to any server.
- The transformer model is downloaded **once** from Hugging Face on first use and cached locally. Resume text is not sent anywhere afterwards.
- E-mails, URLs and phone numbers are removed before scoring. Names, gender, age, photos and other protected attributes are **not** used by the scoring code.
  Free text can still contain such information indirectly (for example a college name or a career gap), so use the *Hide filenames* option and human review.
- Streamlit keeps results in the browser session until you close or refresh the page. If you deploy this app on a shared server, you are responsible for access control and for your legal obligations (GDPR, DPDP Act, etc.).
- This is a portfolio project and has not been validated for real hiring decisions.
        """
    )

# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    method_label = st.radio("Matching model", list(METHOD_LABELS), index=0)
    method = METHOD_LABELS[method_label]
    device = "auto"
    if method == METHOD_TRANSFORMER:
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], help="'auto' uses a GPU if PyTorch can see one.")

    st.subheader("Score weights")
    sim_pct = st.slider("Text similarity weight (%)", 0, 100, 60, 5)
    st.caption(f"Required-skill coverage weight: **{100 - sim_pct}%**")
    st.caption("final = similarity × w_sim + skill coverage × w_skill")

    anonymize = st.checkbox("Hide filenames (show Candidate IDs)", value=False)
    st.caption(f"Max file size: {MAX_FILE_BYTES // (1024 * 1024)} MB · PDF / DOCX only · no OCR")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""

def load_demo_data() -> None:
    """Button callback: runs BEFORE widgets are created on the next rerun, so it may set widget state."""
    st.session_state["jd_text"] = DEMO_JOB_DESCRIPTION
    st.session_state["use_demo_resumes"] = True


left, right = st.columns([1, 1])
with left:
    st.subheader("1. Job description")
    st.text_area("Paste the job description", key="jd_text", height=260, placeholder="Paste the full job description here...")
with right:
    st.subheader("2. Resumes")
    uploaded_files = st.file_uploader("Upload PDF or DOCX resumes", type=["pdf", "docx"], accept_multiple_files=True)
    st.button("Load synthetic demo data (no upload needed)", on_click=load_demo_data)

if uploaded_files:
    st.session_state["use_demo_resumes"] = False

jd_text = st.session_state["jd_text"]
detected_skills = extract_skills(clean_text(jd_text))

st.subheader("3. Required skills")
st.caption("Auto-detected from the job description. Add or remove skills to match what is truly required.")
required_skills = st.multiselect(
    "Required skills",
    options=sorted(SKILL_TAXONOMY),
    default=detected_skills,
    key=f"required_{hash(jd_text)}",
    format_func=lambda s: f"{s}  ·  {CATEGORY_BY_SKILL[s]}",
)
if jd_text.strip() and not required_skills:
    st.warning("No required skills selected - only text similarity will be used for the score.")

analyze = st.button("🔍 Analyze & rank resumes", type="primary")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze:
    st.session_state.pop("analysis", None)
    word_count = len(jd_text.split())
    if word_count < 15:
        st.error("Please paste a job description (at least ~15 words).")
        st.stop()

    documents: list[ParsedDocument] = []
    if st.session_state.get("use_demo_resumes") and not uploaded_files:
        documents = [ParsedDocument(filename=n, text=t) for n, (t, _) in DEMO_RESUMES.items()]
    elif uploaded_files:
        progress = st.progress(0.0, text="Reading resumes...")
        for i, file in enumerate(uploaded_files, start=1):
            documents.append(parse_document(file.name, file.getvalue()))
            progress.progress(i / len(uploaded_files), text=f"Parsed {i}/{len(uploaded_files)}: {file.name}")
        progress.empty()
    else:
        st.error("Upload at least one resume, or click 'Load synthetic demo data'.")
        st.stop()

    failed = [d for d in documents if not d.ok]
    if len(failed) == len(documents):
        st.error("None of the files could be read. See the reasons below.")
        st.table(pd.DataFrame({"file": [d.filename for d in failed], "problem": [d.error for d in failed]}))
        st.stop()

    try:
        matcher = get_matcher(device) if method == METHOD_TRANSFORMER else None
        with st.spinner("Ranking resumes..."):
            results = rank_resumes(
                jd_text,
                documents,
                method=method,
                config=RankingConfig(similarity_weight=sim_pct / 100, skill_weight=(100 - sim_pct) / 100),
                required_skills=required_skills,
                matcher=matcher,
            )
    except Exception as exc:
        st.error(f"Ranking failed: {exc}")
        st.stop()

    st.session_state["analysis"] = {
        "results": results,
        "failed": [(d.filename, d.error) for d in failed],
        "warnings": [(d.filename, w) for d in documents for w in d.warnings],
        "method": method,
        "required": required_skills,
        "total": len(documents),
    }

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
analysis = st.session_state.get("analysis")
if analysis:
    results: list[CandidateResult] = analysis["results"]
    st.divider()
    st.header("Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files uploaded", analysis["total"])
    c2.metric("Ranked", len(results))
    c3.metric("Skipped", len(analysis["failed"]))
    c4.metric("Required skills", len(analysis["required"]))

    if analysis["failed"]:
        with st.expander(f"⚠️ {len(analysis['failed'])} file(s) skipped", expanded=True):
            st.table(pd.DataFrame(analysis["failed"], columns=["file", "problem"]))
    for filename, warning in analysis["warnings"]:
        st.warning(f"{filename}: {warning}")

    table = results_to_dataframe(results, anonymize=anonymize)
    show_cols = ["rank", "candidate", "final_score", "similarity", "skill_coverage", "matched_skills", "missing_skills"]
    st.dataframe(
        table[show_cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "final_score": st.column_config.ProgressColumn("Final score", min_value=0.0, max_value=1.0, format="%.3f"),
            "similarity": st.column_config.ProgressColumn(
                "TF-IDF similarity" if analysis["method"] == METHOD_TFIDF else "Semantic similarity",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "skill_coverage": st.column_config.ProgressColumn("Skill coverage", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )
    st.caption(SCORE_DISCLAIMER)
    st.download_button(
        "⬇️ Download ranking as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="hiresense_ranking.csv",
        mime="text/csv",
    )

    # ----- Detail view -----
    st.subheader("Candidate detail")
    labels = {r.candidate_id if anonymize else r.filename: r for r in results}
    chosen = st.selectbox("Select a candidate", list(labels))
    r = labels[chosen]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rank", f"#{r.rank}")
    m2.metric("Final score", f"{r.final_score:.3f}")
    m3.metric("Text similarity", f"{r.similarity:.3f}")
    m4.metric("Skill coverage", "n/a" if r.skill_coverage is None else f"{r.skill_coverage:.0%}")

    st.markdown("**How the score was computed**")
    st.code(
        f"final = {r.similarity_weight_used:.2f} × similarity ({r.similarity:.3f})"
        + (
            f" + {r.skill_weight_used:.2f} × skill coverage ({r.skill_coverage:.3f})"
            if r.skill_coverage is not None
            else "   (no required skills selected -> similarity only)"
        )
        + f" = {r.final_score:.3f}",
        language="text",
    )
    if r.semantic_similarity is not None:
        st.caption(f"For comparison, the TF-IDF baseline similarity for this resume is {r.tfidf_similarity:.3f}.")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**✅ Matched required skills**")
        st.markdown(" ".join(f":green[`{s}`]" for s in r.matched_skills) or "_None_")
    with s2:
        st.markdown("**❌ Missing required skills**")
        st.markdown(" ".join(f":red[`{s}`]" for s in r.missing_skills) or "_None - all required skills found_")
    if r.extra_skills:
        st.markdown("**➕ Other skills found:** " + ", ".join(f"`{s}`" for s in r.extra_skills))

    with st.expander("📎 Supporting evidence", expanded=True):
        if r.skill_evidence:
            st.markdown("**Where matched skills appear in the resume**")
            for skill, snippets in r.skill_evidence.items():
                for snippet in snippets[:1]:
                    st.markdown(f"- `{skill}`: {snippet}")
        if r.top_terms:
            st.markdown("**Top shared TF-IDF terms with the job description**")
            st.write(", ".join(f"{term} ({weight:.3f})" for term, weight in r.top_terms))
        if r.best_chunks:
            st.markdown("**Resume passages most similar to the job description (semantic)**")
            for text, sim in r.best_chunks:
                st.markdown(f"- (cosine {sim:.3f}) {text[:400]}{'...' if len(text) > 400 else ''}")

    with st.expander("📄 Cleaned resume text (contact details removed)"):
        st.text(r.cleaned_text[:4000] + ("\n..." if len(r.cleaned_text) > 4000 else ""))

    # ----- BiLSTM panel (kept separate from the relevance score) -----
    st.divider()
    with st.expander("🧪 Experimental: BiLSTM resume-category classifier (separate from the score)"):
        predictor, reason = load_bilstm_if_valid()
        if predictor is None:
            st.info(
                f"{reason} Nothing is shown rather than fake predictions. "
                "Train one with `python -m src.training.train_bilstm --csv <file> --text-col <col> --label-col <col>` "
                f"and save it to `{BILSTM_CHECKPOINT}`."
            )
        else:
            st.warning(
                f"This model predicts **{predictor.dataset_info.get('task', 'a label')}** "
                f"({predictor.dataset_info.get('label_meaning', '')}). "
                "It is NOT used in the ranking score and says nothing about candidate suitability."
            )
            predictions = predictor.predict([r.cleaned_text], top_k=3)[0]
            st.table(pd.DataFrame(predictions, columns=["label", "probability"]))
            st.caption(f"Held-out test metrics stored in the checkpoint: {predictor.metrics}")
else:
    st.caption("Add a job description and resumes, then press **Analyze & rank resumes**.")
