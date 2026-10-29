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
RATING_ICON = {"up": "👍", "down": "👎"}
STATUS_ICON = {"complete": "✅", "partial": "🟡", "failed": "❌", "queued": "⏳", "running": "⏳"}
PAGE_SIZE = 20  # history rows per page


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
        invite = st.text_input("Invite code (if you were given one)", key="reg_invite")
        if st.form_submit_button("Create account"):
            try:
                c.register(email, password, invite.strip() or None)
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
    feedback_form(c, job_id, job.get("feedback"))


def feedback_form(c: ScoutClient, job_id: str, current: dict | None) -> None:
    """Thumbs up/down plus an optional comment. Sent together: the API stores both
    (rating again replaces them) and passes the rating to Langfuse as a score."""
    current = current or {}
    with st.form(f"feedback-{job_id}"):
        st.markdown("**Was this report useful?**")
        rating = st.feedback(
            "thumbs",
            key=f"rating-{job_id}",
            default={"up": 1, "down": 0}.get(current.get("rating")),
        )
        comment = st.text_input(
            "Comment (optional)",
            value=current.get("comment") or "",
            max_chars=1000,
            key=f"comment-{job_id}",
        )
        if st.form_submit_button("Send feedback"):
            if rating is None:
                st.warning("Pick 👍 or 👎 first.")
                return
            try:
                c.feedback(job_id, "up" if rating == 1 else "down", comment.strip() or None)
                st.success("Thanks, feedback saved.")
            except ScoutError as exc:
                st.error(md(exc.message))


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


# --- history ---


def _open_job(job_id: str) -> None:
    # A button callback runs before the page is drawn, so it may switch the sidebar page.
    st.session_state["job_id"] = job_id
    st.session_state["page"] = "Ask"


def history_page(c: ScoutClient) -> None:
    st.title("History")
    after = st.session_state.get("history_after")
    try:
        jobs = c.history(limit=PAGE_SIZE, after=after)
    except ScoutError as exc:
        st.error(md(exc.message))
        return
    if not jobs:
        st.info("No older reports." if after else "No research yet. Ask a question first.")
    for job in jobs:
        text_col, open_col = st.columns([5, 1])
        text_col.markdown(f"**{md(job['question'])}**")
        when = job["created_at"][:16].replace("T", " ")
        icon = STATUS_ICON.get(job["status"], "")
        rating = RATING_ICON.get(job.get("rating") or "", "")
        text_col.caption(f"{when} UTC · {icon} {job['status']} · £{job['cost_gbp']:.4f} {rating}")
        open_col.button("Open", key=f"open-{job['id']}", on_click=_open_job, args=(job["id"],))

    newer_col, older_col = st.columns(2)
    if after and newer_col.button("← Newest"):
        st.session_state.pop("history_after")
        st.rerun()
    if len(jobs) == PAGE_SIZE and older_col.button("Older →"):
        st.session_state["history_after"] = jobs[-1]["id"]
        st.rerun()


# --- documents ---

DOC_STATUS = {"processing": "⏳ processing", "ready": "✅ ready", "failed": "❌ failed"}


def documents_page(c: ScoutClient) -> None:
    st.title("My documents")
    st.caption(
        "PDFs you upload are searched when **Search my documents** is ticked. "
        "Only you can see them."
    )
    with st.form("upload", clear_on_submit=True):
        file = st.file_uploader("Add a PDF (max 20 MB)", type=["pdf"])
        if st.form_submit_button("Upload") and file is not None:
            try:
                c.upload(file.name, file.getvalue())
                st.success(md(f"Uploaded {file.name}. Processing it now."))
            except ScoutError as exc:
                st.error(md(exc.message))

    try:
        docs = c.documents()
    except ScoutError as exc:
        st.error(md(exc.message))
        return
    if not docs:
        st.info("No documents yet.")
        return
    for doc in docs:
        line = f"**{md(doc['filename'])}** · {DOC_STATUS.get(doc['status'], doc['status'])}"
        if doc["status"] == "ready":
            line += f" · {doc['pages']} page{'' if doc['pages'] == 1 else 's'}"
        st.markdown(line)
        if doc.get("error"):
            st.caption(md(doc["error"]))
    if any(d["status"] == "processing" for d in docs):
        st.button("Refresh")  # any click reruns the page, which reloads the list


# --- stats ---


def _or_dash(value: float | None, fmt: str) -> str:
    return "–" if value is None else fmt.format(value)


def stats_page(c: ScoutClient) -> None:
    st.title("My usage")
    days = st.selectbox("Period", [7, 30, 90], index=1, format_func=lambda d: f"Last {d} days")
    try:
        s = c.stats(days)
    except ScoutError as exc:
        st.error(md(exc.message))
        return
    if not s["jobs"]:
        st.info("No research in this period.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Reports", s["finished"])
    col2.metric("Avg cost", _or_dash(s["avg_cost_gbp"], "£{:.4f}"))
    col3.metric("Avg time", _or_dash(s["avg_latency_s"], "{:.0f}s"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Failed", _or_dash(s["failure_rate"], "{:.0%}"))
    col2.metric("👍", s["thumbs_up"])
    col3.metric("👎", s["thumbs_down"])

    if per_day := s["reports_per_day"]:
        st.markdown("**Reports per day**")
        st.bar_chart(
            {"day": [d["day"] for d in per_day], "reports": [d["reports"] for d in per_day]},
            x="day",
            y="reports",
        )
