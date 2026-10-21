"""Fact checks for tailored CVs and cover letters, fixed letter parts, PDF rendering and output paths."""

import io
from datetime import date

import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from job_hunter import llm, pdf_render, tailoring
from job_hunter.cv import CvDocument
from job_hunter.tailoring import CoverLetter, TailoredCv


def _role(company: str, role: str, start: str, end: str, bullets: list[str]) -> dict:
    """Tailored-CV role (the source CV's roles also carry employment_type)."""
    return {"company": company, "role": role, "start": start, "end": end, "bullets": bullets}


def _source_role(*args) -> dict:
    return {**_role(*args), "employment_type": None}


SOURCE = CvDocument.model_validate({
    "name": "Jane Doe",
    "headline": "Data & AI Engineer",
    "summary": "I design data platforms and agentic AI systems.",
    "location": "Zagreb, Croatia",
    "work_authorization": "✓ Based in Zagreb — valid residence & work permit",
    "email": "jane@example.com",
    "phone": "+385 1 234 567",
    "links": [{"label": "LinkedIn", "url": "https://www.linkedin.com/in/jane/"}],
    "experience": [
        _source_role("HTEC", "Data Engineer", "Dec 2024", "Ongoing",
                     ["Built a RAG application with hybrid search",
                      "Designed a medallion pipeline across 13 table pairs"]),
        _source_role("Apex Group", "Analytics Engineer", "Sep 2024", "Dec 2024", ["Created DBT models on Snowflake"]),
        _source_role("EPAM Systems", "Data Engineer", "Jul 2022", "Jan 2024",
                     ["Modelled 3 layer delta lake architecture"]),
        _source_role("Artı Maritime", "Business Intelligence Analyst", "Agu2019", "Jan 2022",
                     ["Built Power BI dashboards"]),
        _source_role("Wavin – Orbia", "Project Sales Executive", "Apr2018", "Agu2019",
                     ["Fulfilled +2M Euro sales target"]),
    ],
    "skills": [{"category": "AI", "items": ["RAG", "LangChain"]},
               {"category": "Platforms", "items": ["Databricks", "Snowflake", "DBT"]}],
    "certifications": ["Databricks Data Engineer Professional", "Snowflake SnowPro Core Certification"],
    "education": [{"degree": "MBA", "institution": "Bahçeşehir University", "start": "02/2018", "end": "06/2019",
                   "details": []}],
    "languages": [{"language": "English", "level": "Full Professional Proficiency"}],
    "trainings": [],
    "projects": [],
})

VALID_TAILORED = {
    "summary": "Data & AI engineer building Databricks platforms and RAG applications.",
    "skills": [{"category": "Platforms", "items": ["Databricks", "DBT"]}, {"category": "AI", "items": ["RAG"]}],
    "experience": [
        _role("HTEC", "Data Engineer", "Dec 2024", "Ongoing",
              ["Designed a Databricks medallion pipeline across 13 table pairs", "Built a RAG app with hybrid search"]),
        _role("Apex Group", "Analytics Engineer", "Sep 2024", "Dec 2024", ["Built DBT models on Snowflake"]),
        _role("EPAM Systems", "Data Engineer", "Jul 2022", "Jan 2024", ["Modelled a 3 layer Delta Lake architecture"]),
        _role("Arti Maritime", "Business Intelligence Analyst", "Agu2019", "Jan 2022", []),
        _role("Wavin - Orbia", "Project Sales Executive", "Apr2018", "Agu2019", []),
    ],
    "certifications": ["Databricks Data Engineer Professional"],
}


def _tailored(**overrides) -> dict:
    return {**VALID_TAILORED, **overrides}


def _with_role(index: int, **changes) -> list[dict]:
    roles = [dict(r) for r in VALID_TAILORED["experience"]]
    roles[index] = {**roles[index], **changes}
    return roles


def test_valid_tailored_cv_passes_with_dash_and_case_differences():
    # "Wavin - Orbia" vs "Wavin – Orbia" is the same company; "Arti" vs "Artı" is not.
    data = _tailored(experience=_with_role(3, company="Artı Maritime"))
    TailoredCv.model_validate(data, context={"source": SOURCE})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"experience": _with_role(0, company="Google")}, "company"),
        ({"experience": _with_role(2, start="Jul 2021")}, "start"),
        ({"experience": _with_role(1, role="Lead Analytics Engineer")}, "role"),
        ({"experience": VALID_TAILORED["experience"][:4]}, "all 5 roles"),
        ({"skills": [{"category": "Cloud", "items": ["Kubernetes"]}]}, "skills not in the original CV"),
        ({"certifications": ["AWS Solutions Architect"]}, "certifications not in the original CV"),
        ({"summary": "Cut pipeline costs by 40% for 200 clients."}, "numbers not in the original CV"),
        ({"summary": " ".join(["word"] * 81)}, "max 80"),
    ],
)
def test_tailored_cv_fact_check_rejects(change, message):
    data = _tailored(**change)
    if "experience" not in change:
        data["experience"] = _with_role(3, company="Artı Maritime")
    with pytest.raises(ValidationError, match=message):
        TailoredCv.model_validate(data, context={"source": SOURCE})


def test_tailored_cv_strict_schema_has_no_constraints_leaking():
    schema = llm.strict_schema(TailoredCv)
    assert set(schema["required"]) == {"summary", "skills", "experience", "certifications"}
    assert "maxItems" not in str(schema)


JOB = {"title_en": "Data Engineer", "company": "Acme Data", "city": "Zagreb", "country": "Croatia",
       "description_en": "We need 5+ years with Databricks.", "matching_skills": ["Databricks"],
       "missing_skills": ["Kafka"]}
CLOSING = tailoring.closing_paragraph()


def _letter(paragraphs: list[str], greeting: str = "Dear Hiring Team at Acme Data,") -> dict:
    return {"greeting": greeting, "paragraphs": paragraphs}


def _body(words: int, extra: str = "") -> list[str]:
    filler = " ".join(["platform"] * (words // 2))
    return [f"Acme Data is building a lakehouse {extra} {filler}", f"At HTEC I built a RAG application. {filler}"]


def _context() -> dict:
    return {"source": SOURCE, "job": JOB, "closing": CLOSING}


def test_cover_letter_valid():
    CoverLetter.model_validate(_letter(_body(240)), context=_context())


@pytest.mark.parametrize(
    ("paragraphs", "message"),
    [
        (_body(40), "words"),
        (_body(400), "words"),
        ([p.replace("Acme Data", "your company") for p in _body(240)], "must name the company"),
        (_body(240, extra="saving 35% of costs"), "numbers not in the CV or the job posting"),
    ],
)
def test_cover_letter_fact_check_rejects(paragraphs, message):
    with pytest.raises(ValidationError, match=message):
        CoverLetter.model_validate(_letter(paragraphs, greeting="Dear Hiring Team,"), context=_context())


@pytest.mark.parametrize(
    ("company", "text", "expected"),
    [
        ("Teva Pharmaceuticals", "I want to join Teva because", True),
        ("A.P. Moller - Maersk", "Dear Hiring Team at Maersk,", True),
        ("Woodmark Consulting", "Woodmark's data team", True),
        ("IcruitmenT", "Dear Icruitment team", True),
        ("Woodmark Consulting", "your consulting practice", False),
        ("Teva Pharmaceuticals", "a pharmaceuticals leader", False),
        ("Acme Data", "Acme Data is building", True),
    ],
)
def test_company_mentioned(company, text, expected):
    assert tailoring.company_mentioned(company, text) is expected


def test_cover_letter_allows_numbers_from_job_posting():
    CoverLetter.model_validate(_letter(_body(240, extra="your 5+ years requirement")), context=_context())


def test_closing_paragraph_fixed_text():
    assert CLOSING == (
        "I am based in Zagreb with a valid Croatian residence and work permit, so no sponsorship is needed. "
        "I am available to start quickly. I am also open to B2B cooperation through my own company."
    )


def test_compose_letter_adds_fixed_header_and_sign_off():
    letter = CoverLetter.model_validate(_letter(_body(240)), context=_context())
    document = tailoring.compose_letter(SOURCE, letter, CLOSING, date(2026, 10, 21))
    assert document.contact_line == "jane@example.com | +385 1 234 567 | Zagreb, Croatia | https://www.linkedin.com/in/jane/"
    assert document.date_line == "21 October 2026"
    assert document.text().endswith(f"{CLOSING}\n\nKind regards,\nJane Doe")


def _pdf_text(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)


def test_render_cv_unicode_and_source_facts():
    tailored = TailoredCv.model_validate(_tailored(experience=_with_role(3, company="Artı Maritime")),
                                         context={"source": SOURCE})
    pdf, rendered = pdf_render.render_cv_within_pages(SOURCE, tailored)
    text = _pdf_text(pdf)
    assert pdf_render.page_count(pdf) == 1
    assert "Artı Maritime" in text and "Bahçeşehir" in text
    assert "Wavin – Orbia" in text  # company names always come from the source CV
    assert rendered == tailored


def test_long_cv_is_trimmed_to_two_pages_keeping_recent_roles():
    long_bullet = "Delivered Databricks and Snowflake data pipelines with DBT models and RAG search " * 3
    recent = [dict(r, bullets=["Built Databricks pipelines"] * 6) for r in VALID_TAILORED["experience"][:3]]
    older = [dict(r, bullets=[long_bullet] * 6) for r in VALID_TAILORED["experience"][3:]]
    roles = recent + older
    roles[3]["company"] = "Artı Maritime"
    tailored = TailoredCv.model_validate({**VALID_TAILORED, "experience": roles})
    pdf, rendered = pdf_render.render_cv_within_pages(SOURCE, tailored, max_pages=1)
    assert pdf_render.page_count(pdf) == 1
    assert [len(job.bullets) for job in rendered.experience[:3]] == [6, 6, 6]
    assert sum(len(job.bullets) for job in rendered.experience[3:]) < 12


def test_render_cover_letter_one_page():
    letter = CoverLetter.model_validate(_letter(_body(240)), context=_context())
    document = tailoring.compose_letter(SOURCE, letter, CLOSING, date(2026, 10, 21))
    pdf = pdf_render.render_cover_letter(document)
    assert pdf_render.page_count(pdf) == 1
    assert "Kind regards" in _pdf_text(pdf)


def test_application_dir_is_filesystem_safe():
    path = pdf_render.application_dir("/Volumes/c/s/cv/output/", date(2026, 10, 21), 3, "A.P. Møller – Maersk",
                                      "Senior Data/AI Engineer (m/ž/d)", "123")
    assert path == "/Volumes/c/s/cv/output/2026-10-21/03_A_P_Moller_Maersk_Senior_Data_AI_Engineer_m_z_d_123"
    assert pdf_render.slugify("Inženjer podataka Đakovo") == "Inzenjer_podataka_Dakovo"
    assert pdf_render.slugify(None) == "unknown"
