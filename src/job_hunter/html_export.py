"""Daily application page (index.html + applications.json) for a person or a browser agent.

Each application section has stable data-* attributes and the whole day is embedded as JSON, so an agent
(e.g. Claude in Chrome) can read the apply link, the CV / cover letter file paths and the letter text exactly.
PDF links are relative to the day folder; the local export re-renders the page with absolute local paths,
which a browser upload tool needs.
"""

import html
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from job_hunter.records import ApplyType

INDEX_FILE = "index.html"
JSON_FILE = "applications.json"


@dataclass(frozen=True)
class ApplicationEntry:
    rank: int
    tier: int
    fit_score: int
    job_id: str
    title: str
    company: str | None
    location: str | None
    apply_url: str
    cv_file: str  # relative to the day folder, e.g. "01_OIV_Open_Application_123/cv.pdf"
    cover_letter_file: str
    cover_letter_text: str
    reasons: str
    matching_skills: list[str]
    missing_skills: list[str]
    apply_type: str = ApplyType.UNKNOWN.value  # default keeps older applications.json files readable


# Step-by-step hint for a browser agent; LinkedIn labels depend on the UI language (English / Turkish).
APPLY_INSTRUCTIONS = {
    ApplyType.EXTERNAL: (
        'Open the apply link and click the "Apply" / "Başvur" button. It opens the company\'s own application site '
        "(often in a new tab); continue the application there."
    ),
    ApplyType.EASY_APPLY: (
        'Open the apply link and click "Easy Apply" / "Kolay Başvuru". Fill in the LinkedIn form and upload the CV '
        "when asked."
    ),
    ApplyType.UNKNOWN: (
        'Open the apply link and click the apply button shown ("Apply" / "Başvur" or "Easy Apply" / '
        '"Kolay Başvuru"), then follow the form.'
    ),
}
STOP_BEFORE_SUBMIT = "Upload the CV and cover letter from the paths above when asked, paste the cover letter text " \
    "into text fields, and stop before the final submit so the candidate can review."


def apply_instructions(apply_type: str) -> str:
    try:
        kind = ApplyType(apply_type)
    except ValueError:
        kind = ApplyType.UNKNOWN
    return f"{APPLY_INSTRUCTIONS[kind]} {STOP_BEFORE_SUBMIT}"


def relative_to_day(pdf_path: str, day_dir: str) -> str:
    """'/Volumes/.../output/2026-10-21/01_x/cv.pdf' → '01_x/cv.pdf'."""
    prefix = day_dir.rstrip("/") + "/"
    if not pdf_path.startswith(prefix):
        raise ValueError(f"{pdf_path} is not inside {day_dir}")
    return pdf_path[len(prefix):]


def to_json(day: date, entries: list[ApplicationEntry]) -> str:
    return json.dumps({"date": day.isoformat(), "applications": [asdict(e) for e in entries]}, ensure_ascii=False,
                      indent=2)


def from_json(text: str) -> tuple[date, list[ApplicationEntry]]:
    data = json.loads(text)
    return date.fromisoformat(data["date"]), [ApplicationEntry(**e) for e in data["applications"]]


def render_index(day: date, entries: list[ApplicationEntry], local_dir: Path | None = None) -> str:
    """HTML page for one day. With `local_dir`, file paths are shown and embedded as absolute local paths."""

    def file_path(relative: str) -> str:
        return str((local_dir / relative).resolve()) if local_dir else relative

    records = [
        {**asdict(e), "cv_path": file_path(e.cv_file), "cover_letter_path": file_path(e.cover_letter_file),
         "agent_instructions": apply_instructions(e.apply_type)}
        for e in entries
    ]
    # "</" would end the <script> element early; JSON allows the escaped form.
    embedded = json.dumps({"date": day.isoformat(), "applications": records}, ensure_ascii=False).replace("</", "<\\/")
    sections = "\n".join(_section(e, r["cv_path"], r["cover_letter_path"]) for e, r in zip(entries, records,
                                                                                            strict=True))
    title = f"Job applications {day.isoformat()}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>
  :root {{ color-scheme: light dark; --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --card: #f6f8f9;
          --accent: #17697d; --border: #d9dee1; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #16181a; --fg: #e8eaed; --muted: #a0a6ab; --card: #202326;
          --accent: #5cc3d9; --border: #343a3f; }} }}
  body {{ font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--fg);
         max-width: 960px; margin: 0 auto; padding: 24px 16px; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px; }}
  .application {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px;
                 margin: 16px 0; }}
  .application h2 {{ font-size: 1.15rem; margin: 0 0 4px; }}
  .meta, .muted {{ color: var(--muted); font-size: .9rem; }}
  .badge {{ display: inline-block; border: 1px solid var(--accent); color: var(--accent); border-radius: 999px;
           padding: 0 8px; font-size: .8rem; margin-right: 6px; }}
  dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; margin: 12px 0; }}
  dt {{ font-weight: 600; }} dd {{ margin: 0; overflow-wrap: anywhere; }}
  a {{ color: var(--accent); }}
  textarea {{ width: 100%; min-height: 220px; box-sizing: border-box; font: inherit; color: inherit;
             background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px; }}
  code {{ font-size: .85rem; }}
</style>
</head>
<body>
<h1>{_e(title)}</h1>
<p class="muted">{len(entries)} application(s), best match first. Each section lists the apply link, the CV and
cover letter files to upload, and the cover letter text to paste. Review every form before submitting.</p>
{sections}
<script type="application/json" id="applications-data">{embedded}</script>
<script>
  // Remember the status per job in this browser only.
  for (const select of document.querySelectorAll("select[data-status-for]")) {{
    const key = "job-status-" + select.dataset.statusFor;
    try {{ select.value = localStorage.getItem(key) || "not_applied"; }} catch (e) {{}}
    select.addEventListener("change", () => {{ try {{ localStorage.setItem(key, select.value); }} catch (e) {{}} }});
  }}
</script>
</body>
</html>
"""


def _section(e: ApplicationEntry, cv_path: str, letter_path: str) -> str:
    skills_have = ", ".join(e.matching_skills) or "-"
    skills_missing = ", ".join(e.missing_skills) or "-"
    apply_label = {"external": "company site", "easy_apply": "Easy Apply"}.get(e.apply_type, "apply type unknown")
    return f"""<section class="application" id="job-{_e(e.job_id)}" data-rank="{e.rank}" data-job-id="{_e(e.job_id)}"
  data-apply-url="{_e(e.apply_url)}" data-apply-type="{_e(e.apply_type)}" data-cv-path="{_e(cv_path)}"
  data-cover-letter-path="{_e(letter_path)}">
  <h2>#{e.rank} {_e(e.title)} | {_e(e.company or "")}</h2>
  <div class="meta"><span class="badge">tier {e.tier}</span><span class="badge">fit {e.fit_score}</span>
    <span class="badge">{_e(apply_label)}</span>{_e(e.location or "location not stated")}</div>
  <dl>
    <dt>Apply link</dt><dd><a class="apply-link" href="{_e(e.apply_url)}" target="_blank" rel="noopener">{_e(e.apply_url)}</a></dd>
    <dt>Apply steps</dt><dd class="agent-instructions">{_e(apply_instructions(e.apply_type))}</dd>
    <dt>CV (PDF)</dt><dd><a class="cv-file" href="{_e(e.cv_file)}">{_e(e.cv_file)}</a><br><code class="cv-path">{_e(cv_path)}</code></dd>
    <dt>Cover letter (PDF)</dt><dd><a class="cover-letter-file" href="{_e(e.cover_letter_file)}">{_e(e.cover_letter_file)}</a><br><code class="cover-letter-path">{_e(letter_path)}</code></dd>
    <dt>Why it matches</dt><dd>{_e(e.reasons)}</dd>
    <dt>Matching skills</dt><dd>{_e(skills_have)}</dd>
    <dt>Missing skills</dt><dd>{_e(skills_missing)}</dd>
    <dt>Status</dt><dd><select data-status-for="{_e(e.job_id)}" aria-label="Application status">
      <option value="not_applied">not applied</option><option value="applied">applied</option>
      <option value="skipped">skipped</option></select></dd>
  </dl>
  <label for="letter-{_e(e.job_id)}"><strong>Cover letter text</strong> (for text fields)</label>
  <textarea class="cover-letter-text" id="letter-{_e(e.job_id)}" readonly>{_e(e.cover_letter_text)}</textarea>
</section>"""


def _e(value: str) -> str:
    return html.escape(value, quote=True)
