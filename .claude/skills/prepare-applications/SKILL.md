---
name: prepare-applications
description: Prepare (and, after one batch confirmation, submit) today's job applications in the user's Chrome via Claude in Chrome, using the files synced to Downloads\JobApplications\<date>\, and record each outcome in gold_application_status. Use when the user says "/prepare-applications", "apply to today's jobs" or similar.
---

# Prepare today's applications

Arguments (optional): a date `YYYY-MM-DD` (default today) and `dev` to use the dev schema instead of prod.

## 0. Inputs
1. Day folder: `C:\Users\CihanGirayOner\Downloads\JobApplications\<date>\`. If `applications.json` is missing, run
   `uv run python -m job_hunter.export_applications --date <date> --profile personal-projects` (add
   `--schema dev_cihangiray_oner_medallion` for dev). If it still doesn't exist, report "no applications today" and stop.
2. Read `applications.json` with Python (the browser tools cannot open `file://` pages). Fields per entry:
   rank, title, company, location, apply_type, apply_url, cv_file, cover_letter_file, cover_letter_text.
3. Read `C:\Users\CihanGirayOner\Downloads\JobApplications\applicant_profile.json`: standard answers
   (name, email, phone, address, notice period, salary, demographic defaults...). Never commit or upload this file.
4. Copy each job's PDFs into the git-ignored repo folder `scratch/upload/<rank>/` as `Cihan_Giray_Oner_CV.pdf` and
   `Cihan_Giray_Oner_Cover_Letter.pdf`. The upload tool only accepts files under the working directory.

## 1. Browser (load the Claude in Chrome tools in one ToolSearch call; work in new tabs)
For each entry in rank order, don't stop to ask; collect everything for one summary at the end.
1. Open `apply_url`. Classify from the LinkedIn top card:
   - "Şu an başvuru kabul etmiyor" / "No longer accepting applications" → status **closed**, skip.
   - "Başvuru gönderildi" / "Applied … ago" → status **applied** (already applied), skip.
   - Workplace chip "İş yerinde" / "On-site" and the country is not Croatia → status **not_eligible**, skip.
     Hybrid outside Croatia → not_eligible too. Croatian jobs: any work mode is fine.
2. **external** ("Başvur" / "Apply"): click it; continue on the company site in the new tab, and close the LinkedIn tab.
   **easy_apply** ("Kolay Başvuru" / "Easy Apply"): work through the LinkedIn modal.
3. Fill in the form:
   - Personal and standard questions come from `applicant_profile.json`.
   - Tool or experience questions are answered from the CV (`silver_cv`, or the CV text in the PDF). Default:
     3 years for a tool that is in the CV.
   - Answer **No** for a tool or skill that is not in the CV. Never claim experience the CV doesn't support.
   - Leave optional demographic or diversity questions blank. If one is required, pick "Decline / prefer not to say".
   - Free-text motivation or "tell us about a problem" questions: write from the CV and the cover letter; it's fine
     to extrapolate mildly.
   - Upload the CV. Upload the cover letter when there is a field for it (cover letter or additional documents);
     otherwise paste `cover_letter_text` into a cover-letter text box if there is one.
   - Accept required privacy or data-processing agreements for the application. Leave optional marketing or talent
     pool consents unticked.
4. Uploads:
   - Use `file_upload` on the file input's ref. LinkedIn Easy Apply has no file input until "CV yükle" is clicked.
   - In that case, first run JS that overrides `HTMLInputElement.prototype.click` for `type=file` inputs, so the
     input is appended visibly with `aria-label="intercepted-cv-file-input"` instead of opening the OS picker.
   - Click "CV yükle", `find` the intercepted input, upload, then restore the original `click`.
   - Check that the newly uploaded CV (today's date) is the selected resume.
5. Stop at the **final review / submit step**. Do not click Submit yet. Leave the tab open.
6. Never create accounts, enter passwords or solve CAPTCHAs. If a site requires one, mark the job "needs you" and move on.

## 2. One batch confirmation
Show one table:
- rank, company and title, apply type;
- result: ready / closed / already applied / not eligible / needs you;
- notable answers or risks (e.g. language requirements, travel %, salary field).

Ask once: "Submit the ready ones? (yes / yes except N / no)".

On yes:
- click Submit in each ready tab and confirm the success page (or the LinkedIn "application sent" state);
- report anything that failed validation, fix it and retry once.

## 3. Record statuses (always, including skips)
```
uv run python -m job_hunter.application_status --profile personal-projects --source browser --status <s> --job-ids <id,...> [--notes "..."]
```
Statuses:
- `applied`: submitted, or LinkedIn says already applied.
- `closed`
- `not_eligible`
- `skipped`: the user said no.

Add `--schema dev_cihangiray_oner_medallion` when working on dev data. The job id is the number in `apply_url`.

## 4. Clean up
- Delete `scratch/upload/`.
- Close the tabs you opened.
- Final message: what was submitted, what was recorded, anything the user must do.
