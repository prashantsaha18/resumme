"""
utils/resume_input.py
------------------------
Shared "get resume text" widget used across every ML page (ATS Predictor,
Resume Classification, Skill Recommendation, Resume Analyzer, Job Matching).
Gives the user three consistent ways to provide a resume:
  1. Upload a file (PDF / DOCX / TXT)
  2. Paste text directly
  3. Reuse the last resume text already in session (from Resume Builder
     or a previous upload elsewhere in the app)

Centralizing this avoids each page reinventing (and subtly diverging on)
the same upload logic.
"""

import streamlit as st

from utils.resume_parser import extract_resume_text


def resume_input_widget(key_prefix: str, default_mode: str = "Upload a file") -> str:
    """
    Renders the source-selection UI and returns the resulting resume text
    (empty string if nothing provided yet). `key_prefix` must be unique
    per page to avoid Streamlit widget key collisions.
    """
    has_last = bool(st.session_state.get("last_resume_text", "").strip())
    options = ["Upload a file", "Paste text"]
    if has_last:
        options.append("Use last resume")

    mode = st.radio(
        "Resume source", options, horizontal=True,
        key=f"{key_prefix}_source_mode",
        index=options.index(default_mode) if default_mode in options else 0,
    )

    resume_text = ""

    if mode == "Upload a file":
        uploaded = st.file_uploader(
            "Upload your resume (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            key=f"{key_prefix}_uploader",
        )
        if uploaded is not None:
            try:
                resume_text = extract_resume_text(uploaded)
                if resume_text.strip():
                    st.session_state.last_resume_text = resume_text
                    st.success(f"Loaded '{uploaded.name}' ({len(resume_text.split())} words extracted).")
                else:
                    st.warning("Couldn't extract any text from this file. Try a different format.")
            except Exception as e:
                st.error(f"Failed to parse file: {e}")

    elif mode == "Paste text":
        resume_text = st.text_area(
            "Paste resume text",
            value=st.session_state.get("last_resume_text", ""),
            height=260, key=f"{key_prefix}_paste",
            placeholder="Paste your resume text here...",
        )

    else:  # Use last resume
        resume_text = st.session_state.get("last_resume_text", "")
        with st.expander("Preview loaded resume text"):
            st.text(resume_text[:5000] if resume_text else "(empty)")

    return resume_text
