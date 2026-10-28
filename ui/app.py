"""Scout's web UI.

    streamlit run ui/app.py

Talks to the Scout API at SCOUT_API_URL (default http://localhost:8000). Start the API
first, e.g. with JOBS_MODE=inline on Windows:  uvicorn scout.api.main:create_app --factory
"""

import streamlit as st

from scout.ui import views

st.set_page_config(page_title="Scout", page_icon="🔎", layout="centered")

c = views.client()
if not c.token:
    views.login_page(c)
    st.stop()

with st.sidebar:
    st.markdown(f"Signed in as **{views.md(c.email or '')}**")
    if st.button("Log out"):
        c.logout()
        st.session_state.pop("job_id", None)
        st.rerun()

views.ask_page(c)
