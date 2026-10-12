"""Dictionary-based skill extraction with evidence snippets.

A curated taxonomy (canonical skill -> category + aliases) is matched with
boundary-aware regexes, so 'java' does not match inside 'javascript', 'sql' does
not match inside 'mysql', and 'c++' / 'c#' / '.net' / 'node.js' work correctly.
This is intentionally transparent (no black box): every detected skill can be
traced to a snippet of the resume.
"""

from __future__ import annotations

import re
from functools import lru_cache

# canonical name -> (category, [aliases]). Aliases are matched case-insensitively.
SKILL_TAXONOMY: dict[str, tuple[str, list[str]]] = {
    # Programming languages
    "python": ("Languages", ["python", "python3"]),
    "java": ("Languages", ["java"]),
    "javascript": ("Languages", ["javascript", "js", "ecmascript"]),
    "typescript": ("Languages", ["typescript"]),
    "c++": ("Languages", ["c++"]),
    "c#": ("Languages", ["c#"]),
    "go": ("Languages", ["golang"]),
    "rust": ("Languages", ["rust"]),
    "scala": ("Languages", ["scala"]),
    "kotlin": ("Languages", ["kotlin"]),
    "php": ("Languages", ["php"]),
    "r": ("Languages", ["r programming", "r language", "rstudio"]),
    "matlab": ("Languages", ["matlab"]),
    "bash": ("Languages", ["bash", "shell scripting", "shell script"]),
    # Machine learning / DL
    "machine learning": ("ML/DL", ["machine learning", "ml models"]),
    "deep learning": ("ML/DL", ["deep learning"]),
    "nlp": ("ML/DL", ["nlp", "natural language processing"]),
    "computer vision": ("ML/DL", ["computer vision", "image classification", "object detection"]),
    "pytorch": ("ML/DL", ["pytorch", "torch"]),
    "tensorflow": ("ML/DL", ["tensorflow"]),
    "keras": ("ML/DL", ["keras"]),
    "scikit-learn": ("ML/DL", ["scikit-learn", "scikit learn", "sklearn"]),
    "transformers": ("ML/DL", ["transformers", "transformer models", "bert", "hugging face", "huggingface"]),
    "lstm": ("ML/DL", ["lstm", "bilstm", "bi-lstm", "recurrent neural network", "rnn"]),
    "cnn": ("ML/DL", ["cnn", "convolutional neural network"]),
    "xgboost": ("ML/DL", ["xgboost", "lightgbm", "gradient boosting"]),
    "reinforcement learning": ("ML/DL", ["reinforcement learning"]),
    "llm": ("ML/DL", ["llm", "llms", "large language model", "large language models"]),
    "rag": ("ML/DL", ["rag", "retrieval augmented generation", "retrieval-augmented generation"]),
    "feature engineering": ("ML/DL", ["feature engineering"]),
    "statistics": ("ML/DL", ["statistics", "statistical analysis", "hypothesis testing"]),
    "mlops": ("ML/DL", ["mlops", "mlflow", "model deployment"]),
    # Data
    "pandas": ("Data", ["pandas"]),
    "numpy": ("Data", ["numpy"]),
    "sql": ("Data", ["sql", "postgresql", "mysql", "sqlite", "t-sql", "pl/sql"]),
    "nosql": ("Data", ["nosql", "mongodb", "cassandra", "dynamodb"]),
    "spark": ("Data", ["spark", "pyspark", "apache spark"]),
    "hadoop": ("Data", ["hadoop"]),
    "airflow": ("Data", ["airflow"]),
    "etl": ("Data", ["etl", "data pipelines", "data pipeline"]),
    "tableau": ("Data", ["tableau"]),
    "power bi": ("Data", ["power bi", "powerbi"]),
    "excel": ("Data", ["excel", "microsoft excel"]),
    "data visualization": ("Data", ["data visualization", "data visualisation", "matplotlib", "seaborn", "plotly"]),
    # Web / backend
    "react": ("Web", ["react", "react.js", "reactjs"]),
    "angular": ("Web", ["angular"]),
    "vue": ("Web", ["vue", "vue.js", "vuejs"]),
    "node.js": ("Web", ["node.js", "nodejs"]),
    "django": ("Web", ["django"]),
    "flask": ("Web", ["flask"]),
    "fastapi": ("Web", ["fastapi"]),
    "spring boot": ("Web", ["spring boot", "springboot", "spring framework"]),
    ".net": ("Web", [".net", "dotnet", "asp.net"]),
    "html": ("Web", ["html", "html5"]),
    "css": ("Web", ["css", "css3", "tailwind", "sass"]),
    "rest api": ("Web", ["rest api", "rest apis", "restful", "restful apis"]),
    "graphql": ("Web", ["graphql"]),
    "streamlit": ("Web", ["streamlit"]),
    # DevOps / cloud
    "docker": ("DevOps/Cloud", ["docker", "containerization", "containerisation"]),
    "kubernetes": ("DevOps/Cloud", ["kubernetes", "k8s"]),
    "aws": ("DevOps/Cloud", ["aws", "amazon web services", "sagemaker", "ec2", "s3"]),
    "azure": ("DevOps/Cloud", ["azure"]),
    "gcp": ("DevOps/Cloud", ["gcp", "google cloud"]),
    "terraform": ("DevOps/Cloud", ["terraform"]),
    "ci/cd": ("DevOps/Cloud", ["ci/cd", "cicd", "jenkins", "github actions", "gitlab ci"]),
    "linux": ("DevOps/Cloud", ["linux", "unix"]),
    "git": ("Tools", ["git", "github", "gitlab", "version control"]),
    # Practices
    "agile": ("Practices", ["agile", "scrum", "kanban"]),
    "unit testing": ("Practices", ["unit testing", "unit tests", "pytest", "junit"]),
    "microservices": ("Practices", ["microservices", "microservice"]),
    "data structures": ("Practices", ["data structures", "algorithms"]),
    "oop": ("Practices", ["oop", "object oriented", "object-oriented"]),
}

CATEGORY_BY_SKILL: dict[str, str] = {skill: cat for skill, (cat, _) in SKILL_TAXONOMY.items()}


@lru_cache(maxsize=1)
def _compiled_patterns() -> dict[str, re.Pattern[str]]:
    """Compile one regex per skill. Spaces in aliases match spaces or hyphens."""
    patterns: dict[str, re.Pattern[str]] = {}
    for skill, (_, aliases) in SKILL_TAXONOMY.items():
        parts = []
        for alias in sorted(set(aliases), key=len, reverse=True):
            parts.append(re.escape(alias.lower()).replace(r"\ ", r"[\s\-]+"))
        body = "|".join(parts)
        # Not preceded/followed by a letter or digit => whole-token match.
        patterns[skill] = re.compile(rf"(?<![A-Za-z0-9_])(?:{body})(?![A-Za-z0-9_])", re.IGNORECASE)
    return patterns


def extract_skills(text: str) -> list[str]:
    """Return the sorted list of canonical skills mentioned in ``text``."""
    if not text:
        return []
    return sorted(skill for skill, pattern in _compiled_patterns().items() if pattern.search(text))


def find_skill_evidence(text: str, skill: str, window: int = 90, max_snippets: int = 2) -> list[str]:
    """Return short text snippets around where ``skill`` appears (supporting evidence)."""
    pattern = _compiled_patterns().get(skill)
    if pattern is None or not text:
        return []
    snippets: list[str] = []
    last_end = -1
    for match in pattern.finditer(text):
        if match.start() < last_end:  # avoid overlapping snippets
            continue
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        snippets.append(f"{prefix}{snippet}{suffix}")
        last_end = end
        if len(snippets) >= max_snippets:
            break
    return snippets


def compare_skills(required: list[str], candidate_skills: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (matched, missing, extra) skills.

    ``extra`` = skills the candidate has that the job did not list as required.
    """
    required_set = set(required)
    candidate_set = set(candidate_skills)
    matched = sorted(required_set & candidate_set)
    missing = sorted(required_set - candidate_set)
    extra = sorted(candidate_set - required_set)
    return matched, missing, extra


def skill_coverage(required: list[str], candidate_skills: list[str]) -> float | None:
    """Fraction of required skills the candidate has, or None if nothing is required."""
    if not required:
        return None
    matched, _, _ = compare_skills(required, candidate_skills)
    return len(matched) / len(set(required))
