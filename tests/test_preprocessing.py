import pytest

from src.preprocessing import clean_text, split_into_chunks
from src.skill_extractor import compare_skills, extract_skills, find_skill_evidence, skill_coverage


def test_technical_terms_are_preserved():
    cleaned = clean_text("Skills: C++, C#, Node.js, .NET, scikit-learn, CI/CD")
    for term in ["C++", "C#", "Node.js", ".NET", "scikit-learn", "CI/CD"]:
        assert term in cleaned


def test_contact_details_removed():
    raw = "Contact: jane.doe@example.com | +91 98765 43210 | https://linkedin.com/in/jane | www.site.com"
    cleaned = clean_text(raw)
    assert "@" not in cleaned
    assert "98765" not in cleaned
    assert "linkedin" not in cleaned
    assert "www" not in cleaned


def test_year_ranges_not_mistaken_for_phone_numbers():
    cleaned = clean_text("Worked 2019 - 2022 at Company, then 2022 2023")
    assert "2019" in cleaned and "2022" in cleaned


def test_whitespace_and_bullets_normalised():
    cleaned = clean_text("•  Built   models\r\n\r\n\r\n\r\n●  Shipped\tservice")
    assert "•" not in cleaned
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_hyphenated_line_break_joined():
    assert "machine" in clean_text("mach-\nine learning")


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""


def test_split_into_chunks_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = split_into_chunks(text, max_words=150, overlap=30)
    assert len(chunks) >= 3
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]
    assert chunks[-1].split()[-1] == "w399"


def test_split_short_text_single_chunk():
    assert split_into_chunks("a b c", 150, 30) == ["a b c"]
    assert split_into_chunks("", 150, 30) == []


def test_split_invalid_params():
    with pytest.raises(ValueError):
        split_into_chunks("a b c", max_words=10, overlap=10)


# ---- skill extraction ----
def test_skill_boundaries():
    skills = extract_skills("Experienced in JavaScript and MySQL, no Java.")
    assert "javascript" in skills
    assert "sql" in skills
    assert "java" in skills  # "no Java" is still a mention (dictionary matching is not negation-aware)
    only_js = extract_skills("Expert in JavaScript")
    assert "java" not in only_js


def test_special_character_skills():
    skills = extract_skills("Wrote C++ and C# code, used Node.js and ASP.NET")
    assert {"c++", "c#", "node.js", ".net"} <= set(skills)


def test_aliases_map_to_canonical_name():
    assert "scikit-learn" in extract_skills("Used sklearn for modelling")
    assert "kubernetes" in extract_skills("Managed k8s clusters")
    assert "nlp" in extract_skills("natural language processing projects")


def test_compare_and_coverage():
    matched, missing, extra = compare_skills(["python", "sql", "docker"], ["python", "git"])
    assert matched == ["python"]
    assert missing == ["docker", "sql"]
    assert extra == ["git"]
    assert skill_coverage(["python", "sql", "docker"], ["python", "git"]) == pytest.approx(1 / 3)
    assert skill_coverage([], ["python"]) is None


def test_evidence_snippet_contains_skill():
    text = "Intro text. " * 10 + "Built models with PyTorch for NLP. " + "Outro. " * 10
    snippets = find_skill_evidence(text, "pytorch")
    assert snippets and "PyTorch" in snippets[0]
    assert find_skill_evidence(text, "not-a-skill") == []
