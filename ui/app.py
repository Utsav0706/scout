"""Scout's web UI.

    streamlit run ui/app.py

Talks to the Scout API at SCOUT_API_URL (default http://localhost:8000). For local
development on Windows run `py scripts/dev_api.py` and `py scripts/dev_ui.py` instead.
"""

import streamlit as st

from scout.ui import views

st.set_page_config(page_title="Scout", page_icon="🔎", layout="centered")

PAGES = {
    "Ask": views.ask_page,
    "History": views.history_page,
    "Documents": views.documents_page,
    "Usage": views.stats_page,
}

c = views.client()
if not c.token:
    views.login_page(c)
    st.stop()

with st.sidebar:
    page = st.radio("Go to", list(PAGES), key="page", label_visibility="collapsed")
    st.divider()
    st.caption("Signed in as")
    st.text(c.email or "")  # plain text: markdown would turn the address into a mailto link
    if st.button("Log out"):
        c.logout()
        for key in ("job_id", "history_after"):
            st.session_state.pop(key, None)
        st.rerun()

PAGES[page](c)
