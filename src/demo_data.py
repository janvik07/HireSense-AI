"""Synthetic demo data (fictional, no real people, no personal data).

The relevance grades below were assigned by hand by the project author for a
quick sanity check of the ranking code. They are NOT a benchmark: 8 documents is
far too few to draw conclusions about model quality.
"""

from __future__ import annotations

from pathlib import Path

DEMO_JOB_DESCRIPTION = """Machine Learning Engineer (Junior)

We are looking for a Machine Learning Engineer to build and ship NLP models.

Requirements:
- Strong Python programming skills
- Hands-on experience with PyTorch and scikit-learn
- Experience with NLP and transformers
- Good knowledge of SQL and data handling with pandas
- Familiarity with Docker and Git
- Understanding of machine learning fundamentals and model evaluation

Responsibilities:
- Train, evaluate and deploy text classification and semantic search models
- Work with the team to turn experiments into reliable services
"""

# filename -> (resume text, hand-assigned relevance grade 0..2)
DEMO_RESUMES: dict[str, tuple[str, int]] = {
    "resume_ml_engineer.docx": (
        """SUMMARY
Machine learning engineer focused on NLP and deep learning.

SKILLS
Python, PyTorch, scikit-learn, pandas, SQL, Docker, Git, transformers, NLP

EXPERIENCE
- Built a text classification service with PyTorch and Hugging Face transformers, reaching 0.91 macro-F1 on an internal validation set.
- Wrote data pipelines in pandas and SQL, and packaged models with Docker.
- Tracked experiments in Git and wrote unit tests with pytest.

EDUCATION
B.Tech in Computer Science""",
        2,
    ),
    "resume_mlops.pdf": (
        """SUMMARY
ML engineer with deployment experience.

SKILLS
Python, TensorFlow, scikit-learn, Docker, Kubernetes, AWS, MLflow, Git, SQL

EXPERIENCE
- Deployed machine learning models as REST APIs using FastAPI and Docker on AWS.
- Automated retraining with CI/CD and monitored model drift.
- Built NLP feature pipelines with pandas and SQL.

EDUCATION
M.Sc. in Data Science""",
        2,
    ),
    "resume_data_scientist.docx": (
        """SUMMARY
Data scientist with strong statistics background.

SKILLS
Python, scikit-learn, XGBoost, pandas, NumPy, SQL, statistics, NLP

EXPERIENCE
- Built gradient boosting and logistic regression models for churn prediction.
- Performed text mining and NLP topic analysis on support tickets.
- Presented model evaluation results to stakeholders.

EDUCATION
B.Sc. in Statistics""",
        2,
    ),
    "resume_dl_fresher.pdf": (
        """SUMMARY
Recent graduate interested in deep learning and NLP.

SKILLS
Python, PyTorch, NumPy, LSTM, NLP, Git

PROJECTS
- Trained a BiLSTM sentiment classifier in PyTorch on a public dataset.
- Implemented word embeddings experiments and a small text search tool.

EDUCATION
B.Tech in Information Technology""",
        1,
    ),
    "resume_data_analyst.docx": (
        """SUMMARY
Data analyst experienced in reporting and dashboards.

SKILLS
Excel, SQL, Tableau, Power BI, Python, pandas, statistics

EXPERIENCE
- Built weekly sales dashboards in Tableau and Power BI.
- Cleaned and joined datasets with SQL and pandas.
- Ran A/B tests and hypothesis testing.

EDUCATION
B.Com with analytics certification""",
        1,
    ),
    "resume_backend_java.pdf": (
        """SUMMARY
Backend developer building enterprise services.

SKILLS
Java, Spring Boot, SQL, Docker, REST API, microservices, Git

EXPERIENCE
- Developed REST APIs with Spring Boot and PostgreSQL.
- Containerised services with Docker and set up Jenkins pipelines.

EDUCATION
B.E. in Computer Engineering""",
        0,
    ),
    "resume_frontend.docx": (
        """SUMMARY
Frontend developer creating responsive interfaces.

SKILLS
JavaScript, React, TypeScript, HTML, CSS, Git

EXPERIENCE
- Built reusable React components and a design system.
- Improved page load times with code splitting.

EDUCATION
BCA""",
        0,
    ),
    "resume_devops.pdf": (
        """SUMMARY
DevOps engineer automating cloud infrastructure.

SKILLS
AWS, Terraform, Kubernetes, Docker, Linux, Bash, Jenkins, Python scripting

EXPERIENCE
- Managed Kubernetes clusters and Terraform modules on AWS.
- Wrote Python and Bash automation for deployments and monitoring.

EDUCATION
B.Tech in Electronics""",
        0,
    ),
}


def demo_relevance() -> dict[str, int]:
    """filename -> hand-assigned relevance grade (0, 1 or 2)."""
    return {name: grade for name, (_, grade) in DEMO_RESUMES.items()}


def write_demo_files(output_dir: str | Path) -> list[Path]:
    """Write the synthetic resumes as real .docx / .pdf files plus jd.txt and relevance.csv.

    Handy for testing the upload flow in the Streamlit app.
    """
    import fitz  # PyMuPDF
    from docx import Document

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    (out / "job_description.txt").write_text(DEMO_JOB_DESCRIPTION, encoding="utf-8")
    written.append(out / "job_description.txt")

    for filename, (text, _) in DEMO_RESUMES.items():
        path = out / filename
        if filename.endswith(".docx"):
            document = Document()
            for line in text.split("\n"):
                document.add_paragraph(line)
            document.save(path)
        else:
            pdf = fitz.open()
            lines = text.split("\n")
            for start in range(0, len(lines), 45):
                page = pdf.new_page()
                page.insert_textbox(fitz.Rect(50, 50, 545, 790), "\n".join(lines[start : start + 45]), fontsize=10)
            pdf.save(path)
            pdf.close()
        written.append(path)

    relevance_lines = ["filename,relevance_grade"] + [f"{n},{g}" for n, g in demo_relevance().items()]
    (out / "relevance.csv").write_text("\n".join(relevance_lines) + "\n", encoding="utf-8")
    written.append(out / "relevance.csv")
    return written
