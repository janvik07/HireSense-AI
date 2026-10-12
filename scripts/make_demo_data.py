"""Write the synthetic demo resumes (.docx/.pdf), job_description.txt and relevance.csv.

    python scripts/make_demo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.demo_data import write_demo_files  # noqa: E402

if __name__ == "__main__":
    out_dir = Path("data/demo")
    files = write_demo_files(out_dir)
    print(f"Wrote {len(files)} files to {out_dir.resolve()}:")
    for path in files:
        print("  -", path.name)
