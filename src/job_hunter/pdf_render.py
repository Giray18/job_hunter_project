"""PDF rendering for tailored CVs (max two pages) and cover letters (one page)."""

import io
import re
import unicodedata
from datetime import date
from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader

from job_hunter.cv import CvDocument
from job_hunter.tailoring import LetterDocument, TailoredCv, TailoredExperience, contact_line

FONT_DIR = Path(__file__).parent / "fonts"
FONT = "DejaVu"
ACCENT = (23, 105, 125)
MARGIN = 16
MAX_CV_PAGES = 2
# Bullets of the most recent roles are never trimmed to make the CV fit.
PROTECTED_ROLES = 3


class _Pdf(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.add_font(FONT, "", str(FONT_DIR / "DejaVuSans.ttf"))
        self.add_font(FONT, "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self.add_page()

    def text_block(self, text: str, size: float = 9.5, bold: bool = False, height: float = 4.8, indent: float = 0):
        self.set_font(FONT, "B" if bold else "", size)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")

    def section(self, title: str):
        self.ln(2.5)
        self.set_font(FONT, "B", 11)
        self.set_text_color(*ACCENT)
        self.cell(0, 6, title.upper(), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ACCENT)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_text_color(0, 0, 0)
        self.ln(1.5)

    def bullet(self, text: str):
        self.set_font(FONT, "", 9.5)
        self.set_x(self.l_margin + 3)
        self.cell(4, 4.8, "•")
        self.multi_cell(0, 4.8, text, new_x="LMARGIN", new_y="NEXT")


def render_cv(source: CvDocument, tailored: TailoredCv) -> bytes:
    pdf = _Pdf()
    pdf.set_font(FONT, "B", 20)
    pdf.cell(0, 9, source.name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*ACCENT)
    pdf.text_block(source.headline, size=12, bold=True, height=6)
    pdf.set_text_color(80, 80, 80)
    pdf.text_block(contact_line(source), size=8.5)
    if source.work_authorization:
        pdf.text_block(source.work_authorization.lstrip("✓ ").strip(), size=8.5)
    pdf.set_text_color(0, 0, 0)

    pdf.section("Profile")
    pdf.text_block(tailored.summary)

    pdf.section("Skills")
    for group in tailored.skills:
        if group.items:
            pdf.text_block(f"{group.category}: {', '.join(group.items)}")

    pdf.section("Experience")
    for job, original in zip(tailored.experience, source.experience, strict=True):
        # Company, role and dates always come from the source CV.
        kind = f" ({original.employment_type})" if original.employment_type else ""
        pdf.ln(1)
        pdf.text_block(f"{original.role} | {original.company}{kind}", size=10, bold=True, height=5.2)
        pdf.set_text_color(90, 90, 90)
        pdf.text_block(f"{original.start} - {original.end}", size=8.5, height=4.4)
        pdf.set_text_color(0, 0, 0)
        for b in job.bullets:
            pdf.bullet(b)

    if tailored.certifications:
        pdf.section("Certifications")
        pdf.text_block(" · ".join(tailored.certifications))

    pdf.section("Education")
    for edu in source.education:
        dates = f" ({edu.start} - {edu.end})" if edu.start or edu.end else ""
        pdf.text_block(f"{edu.degree}, {edu.institution}{dates}")

    if source.languages:
        pdf.section("Languages")
        pdf.text_block(", ".join(f"{lang.language} ({lang.level})" for lang in source.languages))
    return bytes(pdf.output())


def render_cv_within_pages(source: CvDocument, tailored: TailoredCv,
                           max_pages: int = MAX_CV_PAGES) -> tuple[bytes, TailoredCv]:
    """Render; while over `max_pages`, drop the last bullet of the oldest role that still has bullets
    (never touching the most recent roles). Returns the PDF and the CV as rendered."""
    while True:
        pdf = render_cv(source, tailored)
        if page_count(pdf) <= max_pages:
            return pdf, tailored
        trimmed = _drop_one_old_bullet(tailored)
        if trimmed is None:
            return pdf, tailored
        tailored = trimmed


def _drop_one_old_bullet(tailored: TailoredCv) -> TailoredCv | None:
    experience = list(tailored.experience)
    for i in range(len(experience) - 1, PROTECTED_ROLES - 1, -1):
        if experience[i].bullets:
            experience[i] = TailoredExperience(**{**experience[i].model_dump(), "bullets": experience[i].bullets[:-1]})
            return tailored.model_copy(update={"experience": experience})
    return None


def render_cover_letter(letter: LetterDocument) -> bytes:
    pdf = _Pdf()
    pdf.set_font(FONT, "B", 16)
    pdf.cell(0, 8, letter.name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(80, 80, 80)
    pdf.text_block(letter.contact_line, size=8.5)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)
    pdf.text_block(letter.date_line, size=10, height=5.5)
    pdf.ln(4)
    for paragraph in [letter.greeting, *letter.paragraphs, letter.closing, letter.sign_off]:
        pdf.text_block(paragraph, size=10.5, height=5.6)
        pdf.ln(3)
    return bytes(pdf.output())


def page_count(pdf_bytes: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)


# Letters that Unicode decomposition does not reduce to ASCII.
_TRANSLITERATE = str.maketrans({"đ": "d", "Đ": "D", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "ß": "ss", "æ": "ae",
                                "Æ": "AE", "ı": "i", "œ": "oe"})


def slugify(text: str | None, max_length: int = 40) -> str:
    ascii_text = unicodedata.normalize("NFKD", (text or "").translate(_TRANSLITERATE))
    ascii_text = ascii_text.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    return slug[:max_length].rstrip("_") or "unknown"


def application_dir(output_root: str, day: date, rank: int, company: str | None, title: str, job_id: str) -> str:
    """e.g. /Volumes/.../cv/output/2026-10-21/01_Teva_Pharmaceuticals_Business_Data_Steward_4479821741"""
    return f"{output_root.rstrip('/')}/{day.isoformat()}/{rank:02d}_{slugify(company)}_{slugify(title)}_{job_id}"
