"""Job hunter chat: ask questions about the pipeline's jobs (RAG over the Vector Search index) and browse them."""

import os

import streamlit as st
from backend import Backend
from chat_logic import NO_STATUS, Filters

st.set_page_config(page_title="Job hunter chat", layout="wide")

COUNTRIES = ["HR", "SI", "RS", "BA", "ME", "MK", "AL", "AT", "HU", "IT", "DE", "NL", "BE", "FR", "ES", "PT", "IE", "GB",
             "PL", "CZ", "SK", "RO", "BG", "GR", "CY", "MT", "SE", "DK", "NO", "FI", "EE", "LV", "LT", "LU", "CH"]
WORK_MODES = ["remote", "hybrid", "onsite", "unknown"]
SENIORITIES = ["intern", "junior", "mid", "senior", "lead", "principal", "unknown"]
STATUSES = [NO_STATUS, "applied", "skipped", "closed", "not_eligible"]
TABLE_COLUMNS = ["fit_score", "company", "title_en", "country_code", "city", "work_mode", "seniority", "status",
                 "first_seen_at", "url"]


@st.cache_resource
def backend() -> Backend:
    return Backend()


def sidebar_filters() -> Filters:
    st.sidebar.header("Filters")
    return Filters(
        country_codes=st.sidebar.multiselect("Country", COUNTRIES),
        work_modes=st.sidebar.multiselect("Work mode", WORK_MODES),
        seniorities=st.sidebar.multiselect("Seniority", SENIORITIES),
        statuses=st.sidebar.multiselect("Application status", STATUSES, help=f"'{NO_STATUS}' = not handled yet"),
        min_fit=st.sidebar.slider("Minimum fit score", 0, 100, 0, step=5),
        max_age_days=st.sidebar.slider("First seen within (days)", 1, 90, 30),
    )


def job_table(rows: list[dict], key: str):
    if not rows:
        st.info("No jobs match.")
        return
    st.dataframe(
        [{c: r.get(c) for c in TABLE_COLUMNS} for r in rows],
        column_config={"url": st.column_config.LinkColumn("LinkedIn", display_text="open"),
                       "fit_score": st.column_config.NumberColumn("Fit"),
                       "first_seen_at": st.column_config.DatetimeColumn("First seen", format="YYYY-MM-DD")},
        hide_index=True, use_container_width=True, key=f"{key}-table",
    )
    documents(rows, key)


def documents(rows: list[dict], key: str):
    """Download buttons for the tailored CV and cover letter of a job that has them."""
    with_docs = {f"{r['company']} - {r['title_en']}": r for r in rows if r.get("cv_pdf_path")}
    if not with_docs:
        return
    choice = st.selectbox("Tailored documents", list(with_docs), index=None, placeholder="Pick a job",
                          key=f"{key}-docs")
    if choice:
        job = with_docs[choice]
        left, right = st.columns(2)
        for column, path_key, label in ((left, "cv_pdf_path", "CV"), (right, "cover_letter_pdf_path", "Cover letter")):
            if job.get(path_key):
                column.download_button(f"Download {label}", backend().download(job[path_key]),
                                       file_name=os.path.basename(job[path_key]), mime="application/pdf",
                                       key=f"{key}-{path_key}")


def chat_tab(filters: Filters):
    history = st.session_state.setdefault("history", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("e.g. Remote analytics engineer jobs with dbt and a fit above 70?")
    if not question:
        if st.session_state.get("last_rows"):
            with st.expander("Jobs behind the last answer", expanded=False):
                job_table(st.session_state["last_rows"], "last")
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Searching jobs..."):
        rows = backend().search(question, filters)
        answer = backend().answer(question, rows, history[-6:])
        st.markdown(answer)
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    st.session_state["last_rows"] = rows
    with st.expander(f"Jobs behind this answer ({len(rows)})", expanded=True):
        job_table(rows, "answer")


def browse_tab(filters: Filters):
    rows = backend().jobs(filters)
    st.caption(f"{len(rows)} job(s), best fit first")
    job_table(rows, "browse")


filters = sidebar_filters()
st.title("Job hunter")
chat, browse = st.tabs(["Chat", "Browse"])
with chat:
    chat_tab(filters)
with browse:
    browse_tab(filters)
