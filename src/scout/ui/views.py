"""Streamlit page views. Kept apart from ui/app.py so pages can be tested with AppTest
and the API calls stay in ScoutClient."""

from __future__ import annotations

import re

import streamlit as st

from scout.ui.client import ScoutClient, ScoutError

VERDICT_BADGE = {
    "supported": "✅",
    "unsupported": "⚠️ not supported by source",
    "contradicted": "❌ contradicted by source",
    "unverified": "❔ not checked",
}
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#$~|])")


def md(text: str) -> str:
    """Escape model-written text for st.markdown: no formatting, links or HTML from the
    model, and no "$...$" turning prices into maths."""
    return _MD_SPECIAL.sub(r"\\\1", text or "")


def client() -> ScoutClient:
    if "client" not in st.session_state:
        st.session_state["client"] = ScoutClient()
    return st.session_state["client"]


# --- login ---


def login_page(c: ScoutClient) -> None:
    st.title("🔎 Scout")
    st.caption("Research questions answered with cited, fact-checked reports.")
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab, st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            try:
                c.login(email, password)
                st.rerun()
            except ScoutError as exc:
                st.error(exc.message)
    with register_tab, st.form("register"):
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password (8+ characters)", type="password", key="reg_pw")
        if st.form_submit_button("Create account"):
            try:
                c.register(email, password)
                c.login(email, password)
                st.rerun()
            except ScoutError as exc:
                st.error(exc.message)


# --- ask ---


def ask_page(c: ScoutClient) -> None:
    st.title("Ask a research question")
    with st.form("ask"):
        question = st.text_area("Question", placeholder="e.g. Why did UK inflation rise in 2026?")
        col1, col2 = st.columns(2)
        use_web = col1.checkbox("Search the web", value=True)
        use_docs = col2.checkbox("Search my documents", value=True)
        submitted = st.form_submit_button("Research")
    if submitted:
        try:
            job_id = c.start_research(question, use_web=use_web, use_docs=use_docs)
        except ScoutError as exc:
            st.error(exc.message)
            return
        follow_progress(c, job_id)
        st.session_state["job_id"] = job_id
    if job_id := st.session_state.get("job_id"):
        report_view(c, job_id)


def follow_progress(c: ScoutClient, job_id: str) -> None:
    """Show each step as it happens (server-sent events), until the job is done."""
    with st.status("Researching…", expanded=True) as status:
        try:
            for event, data in c.stream(job_id):
                if event == "progress":
                    st.write(md(data["message"]))
                elif event in ("done", "timeout"):
                    final = data.get("status", "done")
                    status.update(
                        label=f"Finished: {final}",
                        state="complete" if final == "complete" else "error",
                        expanded=False,
                    )
        except ScoutError as exc:
            status.update(label=f"Lost the progress stream: {exc.message}", state="error")


# --- report ---


def report_view(c: ScoutClient, job_id: str) -> None:
    try:
        job = c.job(job_id)
    except ScoutError as exc:
        st.error(exc.message)
        return
    st.subheader(md(job["question"]))
    if job["status"] in ("queued", "running"):
        st.info("Still researching. This page shows the report when it's ready.")
        return
    report = job.get("report")
    if not report:
        st.error(md(job.get("error") or "This research job failed."))
        return
    render_report(report)


def _source_line(src: dict) -> str:
    target = src["url_or_doc"]
    title = md(src["title"])
    # Only real web links are clickable; documents show their name and page.
    if target.startswith(("http://", "https://")):
        return f"**[{src['id']}]** [{title}]({target})"
    return f"**[{src['id']}]** {title} ({md(target)})"


def render_report(report: dict) -> None:
    if report["status"] != "complete":
        st.warning(md(f"Partial report: {report.get('stop_reason') or report['status']}"))
    st.markdown(md(report["summary"]))
    sources = {s["id"]: s for s in report.get("sources", [])}

    for section in report.get("sections", []):
        st.markdown(f"#### {md(section['heading'])}")
        for i, claim in enumerate(section["claims"]):
            text_col, src_col = st.columns([5, 1])
            badge = VERDICT_BADGE.get(claim["verdict"], "")
            refs = " ".join(f"[{sid}]" for sid in claim["source_ids"])
            text_col.markdown(f"{md(claim['text'])} {md(refs)} {badge}")
            with src_col.popover("Sources", key=f"src-{section['heading']}-{i}"):
                for sid in claim["source_ids"]:
                    if sid in sources:
                        st.markdown(_source_line(sources[sid]))
                if claim.get("quote"):
                    st.caption("What the source says (checked word for word):")
                    st.markdown(f"> {md(claim['quote'])}")

    if report.get("limitations"):
        with st.expander("Limitations"):
            for item in report["limitations"]:
                st.markdown(f"- {md(item)}")
    if sources:
        with st.expander("Sources", expanded=False):
            for src in sources.values():
                st.markdown(_source_line(src))

    v = report.get("verification") or {}
    llm_calls = sum(1 for s in report.get("steps", []) if s["kind"] == "llm")
    parts = [
        f"£{report['cost_gbp']:.4f}",
        f"{report['latency_s']:.0f}s",
        f"{llm_calls} model calls",
    ]
    if v.get("support_rate") is not None:
        parts.append(
            f"{v['supported']}/{v['checked'] - v['unverified']} claims supported on first check"
        )
    if report.get("trace_id"):
        parts.append(f"trace {report['trace_id'][:12]}…")
    st.caption(" · ".join(parts))
