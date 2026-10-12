"""Dependency smoke test. Run from the project root:

    python scripts/smoke_test.py

Checks that every library imports, prints versions, and runs tiny end-to-end checks
(PDF + DOCX parsing in memory, TF-IDF ranking). It does NOT download any model.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGES = [
    ("streamlit", "streamlit"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("scipy", "scipy"),
    ("PyMuPDF", "fitz"),
    ("python-docx", "docx"),
    ("torch", "torch"),
    ("sentence-transformers", "sentence_transformers"),
    ("pytest", "pytest"),
]


def check_imports() -> bool:
    ok = True
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  [WARN] Python 3.10+ is required (3.12 recommended).")
        ok = False
    for display, module_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", getattr(module, "VersionBind", "?"))
            print(f"  [ OK ] {display:24s} {version}")
        except Exception as exc:
            print(f"  [FAIL] {display:24s} {type(exc).__name__}: {exc}")
            ok = False
    try:
        import torch

        print(f"  CUDA available: {torch.cuda.is_available()} (CPU is enough for this project)")
    except Exception:
        pass
    return ok


def check_pipeline() -> bool:
    try:
        from src.demo_data import DEMO_JOB_DESCRIPTION, DEMO_RESUMES
        from src.parser import ParsedDocument
        from src.ranking import rank_resumes

        docs = [ParsedDocument(filename=n, text=t) for n, (t, _) in DEMO_RESUMES.items()]
        results = rank_resumes(DEMO_JOB_DESCRIPTION, docs)
        print(f"  [ OK ] TF-IDF ranking ran on {len(results)} synthetic resumes. Top: {results[0].filename}")
        return True
    except Exception as exc:
        print(f"  [FAIL] pipeline check: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    imports_ok = check_imports()
    pipeline_ok = check_pipeline()
    print("\nRESULT:", "all checks passed" if imports_ok and pipeline_ok else "some checks failed")
    raise SystemExit(0 if imports_ok and pipeline_ok else 1)
