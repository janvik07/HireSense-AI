# HireSense AI

Explainable resume screening and job matching: TF-IDF baseline, Sentence-Transformer semantic matching,
and a separate PyTorch BiLSTM text classifier. Built as a fresher ML/DL portfolio project.

> **Important:** every score is a *relevance indicator* (text similarity + required-skill coverage).
> It is **not** a hiring probability and must not be the only basis for a hiring decision.

## Features
- Upload one job description + many PDF/DOCX resumes (PyMuPDF, python-docx)
- File validation: unsupported type, empty, oversized, corrupt, password-protected, scanned/no-text files are reported, never crash the batch
- Cleaning that keeps `C++`, `C#`, `Node.js`, `.NET`, `scikit-learn`; removes e-mail/URL/phone
- Dictionary skill extraction with matched / missing skills and evidence snippets
- Ranking with a transparent formula: `final = w_sim × similarity + w_skill × required-skill coverage`
- Models: **TF-IDF + cosine** baseline, **all-MiniLM-L6-v2** (pretrained, *not* fine-tuned) with long-document chunking, optional GPU
- Streamlit dashboard, per-candidate detail view, CSV export, privacy notice, "hide filenames" option
- Evaluation script (precision@k, nDCG@k, Spearman, TF-IDF vs transformer), pytest tests

## Setup (Windows 11, PowerShell, Python 3.12)
```powershell
cd HireSense-AI
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # smaller CPU build
python -m pip install -r requirements.txt
python scripts\smoke_test.py          # dependency smoke test
```

## Run
```powershell
python scripts\make_demo_data.py      # optional: writes synthetic .docx/.pdf resumes to data\demo
streamlit run app.py                  # then click "Load synthetic demo data" or upload files
python -m pytest                      # unit tests
python -m src.evaluation --skip-transformer   # TF-IDF only
python -m src.evaluation              # + transformer (downloads ~90 MB model on first run)
```

## Train the BiLSTM
```powershell
# pipeline check on SYNTHETIC data (checkpoint flagged is_demo, hidden in the app)
python -m src.training.train_bilstm --demo --epochs 5

# real data: any CSV with a text column and a label column (e.g. a Kaggle resume-category CSV)
python -m src.training.train_bilstm --csv data\raw\resumes.csv --text-col Resume --label-col Category
```
The script prints the label distribution first. A "Category" label means **resume category**, so the model
predicts a category - it does **not** measure suitability for a job. The app shows BiLSTM output only
when `models_saved/bilstm_resume_category.pt` exists, was not trained on demo data and has its label mapping,
and always in a separate panel, never inside the ranking score. Use `--labels-are-relevance` only if your
CSV truly contains resume-job relevance labels.

## Project layout
```
app.py                       Streamlit dashboard
src/parser.py                validation + PDF/DOCX text extraction
src/preprocessing.py         cleaning + chunking
src/skill_extractor.py       skill taxonomy, matching, evidence
src/ranking.py               TF-IDF, score combination, ranking, CSV table
src/evaluation.py            metrics + method comparison (CLI)
src/demo_data.py             synthetic JD/resumes (+ hand-assigned grades for a sanity check)
src/models/transformer_matcher.py   sentence-transformer matcher (chunking, caching, device)
src/models/bilstm.py         vocabulary, dataset, BiLSTM, checkpoints, predictor
src/training/train_bilstm.py training / validation / test script
tests/                       pytest tests
scripts/                     smoke_test.py, make_demo_data.py
```

## Limitations and privacy
- Skill extraction is dictionary-based: it is not negation-aware ("no Java" still matches) and only knows the taxonomy in `skill_extractor.py`.
- TF-IDF IDF values are fitted on the uploaded batch, so scores are comparable only within one run.
- No OCR: scanned PDFs are reported as unreadable.
- Resume text is processed in memory locally; the transformer model is downloaded once from Hugging Face. Names are not used by the code, but free text can still carry indirect signals (college, gaps, location). Add human review, bias auditing and legal review before any real use.
- The demo relevance grades are hand-assigned on 8 synthetic resumes. They are a code sanity check, not a benchmark.

## Code kaise kaam karta hai (Hinglish)
1. `parser.py` file ko bytes se padhta hai, check karta hai ki PDF/DOCX sahi hai ya nahi, aur text nikalta hai. Koi file kharab ho to sirf uska error dikhata hai, baaki resumes chalte rehte hain.
2. `preprocessing.py` text saaf karta hai (email/phone hatata hai) lekin `C++`, `Node.js` jaise technical words nahi todta.
3. `skill_extractor.py` skills ki list se resume aur JD match karta hai - isse matched aur missing skills milti hain, saath me evidence line.
4. `ranking.py` TF-IDF cosine similarity nikalta hai, phir `similarity` aur `skill coverage` ko weights ke saath jodta hai. Formula screen par dikhta hai, koi black box nahi.
5. `transformer_matcher.py` lamba resume chhote chunks me todta hai, har chunk ka embedding banata hai, aur JD se sabse similar chunks ka average leta hai. Model pretrained hai - fine-tune nahi kiya gaya.
6. `bilstm.py` + `train_bilstm.py` ek alag PyTorch classifier hai (tokenize -> embedding -> BiLSTM -> linear). Ye ranking score me mix nahi hota.
