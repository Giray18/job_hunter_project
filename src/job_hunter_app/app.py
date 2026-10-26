"""Job hunter chat: ask questions about the pipeline's jobs (RAG over the Vector Search index), search the index
directly and browse the jobs."""

import os
import time

import chat_logic
import streamlit as st
from backend import Backend
from chat_logic import NO_STATUS, Filters
from databricks.sdk.errors import DatabricksError

st.set_page_config(page_title="Job hunter chat", layout="wide")

COUNTRIES = ["HR", "SI", "RS", "BA", "ME", "MK", "AL", "AT", "HU", "IT", "DE", "NL", "BE", "FR", "ES", "PT", "IE", "GB",
             "PL", "CZ", "SK", "RO", "BG", "GR", "CY", "MT", "SE", "DK", "NO", "FI", "EE", "LV", "LT", "LU", "CH"]
WORK_MODES = ["remote", "hybrid", "onsite", "unknown"]
SENIORITIES = ["intern", "junior", "mid", "senior", "lead", "principal", "unknown"]
STATUSES = [NO_STATUS, "applied", "skipped", "closed", "not_eligible"]
SEARCH_COLUMNS = ["search_rank", "search_score"]
SEARCH_MODES = {"Hybrid (vector + keywords)": "HYBRID", "Vector only (ANN)": "ANN"}
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
    columns = (SEARCH_COLUMNS if "search_rank" in rows[0] else []) + TABLE_COLUMNS
    st.dataframe(
        [{c: r.get(c) for c in columns} for r in rows],
        column_config={"search_rank": st.column_config.NumberColumn("#"),
                       "search_score": st.column_config.NumberColumn(
                           "Index score", format="%.4f",
                           help="Relevance from the Vector Search index; higher is closer. Compare within one search "
                                "only: ANN scores come from the vector distance, Hybrid scores fuse vector and "
                                "keyword ranks."),
                       "url": st.column_config.LinkColumn("LinkedIn", display_text="open"),
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


def semantic_tab(filters: Filters):
    """Query the Vector Search index directly and show its raw ranking."""
    st.caption(f"Searches `{backend().index_name}`: the query is embedded with `{backend().embedding_endpoint}` "
               "(the model that embedded the jobs) and matched against the job vectors in the index.")
    with st.form("semantic"):
        query = st.text_input("Search", placeholder="e.g. lakehouse platform with dbt and Airflow")
        left, right = st.columns(2)
        mode = left.radio("Mode", list(SEARCH_MODES), horizontal=True)
        num_results = right.slider("Results", 5, 50, 15, step=5)
        submitted = st.form_submit_button("Search the index")
    if submitted and query:
        started = time.perf_counter()
        hits = backend().query_index(query, filters, SEARCH_MODES[mode], num_results)
        index_ms = (time.perf_counter() - started) * 1000
        rows = chat_logic.with_hits(backend().jobs(filters, [h["job_id"] for h in hits]), hits)
        st.session_state["semantic"] = {"hits": hits, "rows": rows, "mode": SEARCH_MODES[mode], "ms": index_ms,
                                        "filters": chat_logic.index_filters_json(filters)}
    result = st.session_state.get("semantic")
    if result:
        st.caption(f"`query_type={result['mode']}` · filters sent to the index: `{result['filters'] or 'none'}` · "
                   f"{len(result['hits'])} hit(s) in {result['ms']:.0f} ms (embedding + index query) · "
                   f"{len(result['rows'])} left after the fit/status/age filters")
        job_table(result["rows"], "semantic")
        with st.expander("Raw index response"):
            st.dataframe(result["hits"], hide_index=True, use_container_width=True)
    with st.expander("About the index"):
        try:
            st.json(backend().index_info())
        except DatabricksError as error:  # describing the index needs more than the SELECT used for searching
            st.warning(f"Index details unavailable: {error}")


def browse_tab(filters: Filters):
    rows = backend().jobs(filters)
    st.caption(f"{len(rows)} job(s), best fit first")
    job_table(rows, "browse")


filters = sidebar_filters()
st.title("Job hunter")
chat, semantic, browse = st.tabs(["Chat", "Semantic search", "Browse"])
with chat:
    chat_tab(filters)
with semantic:
    semantic_tab(filters)
with browse:
    browse_tab(filters)
