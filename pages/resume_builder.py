"""
pages/resume_builder.py
-------------------------
Resume Builder page: collects structured resume data via forms, lets the
user pick a template (Modern / Professional / Minimal), generates an
ATS-friendly PDF, and offers a one-click "Analyze this resume" hand-off
into the ML pages using the flattened resume text.
"""

import streamlit as st

from utils.database import save_resume_record
from utils.pdf_generator import ResumeData, generate_resume_pdf, resume_data_to_plain_text


def _init_list_state(key, default_count=1, template=None):
    if key not in st.session_state:
        st.session_state[key] = [dict(template or {}) for _ in range(default_count)]


def render(models):
    st.markdown("## 📝 Resume Builder")
    st.caption("Fill in your details, choose a template, and generate an ATS-friendly PDF resume.")

    _init_list_state("edu_entries", 1, {"degree": "", "school": "", "year": "", "details": ""})
    _init_list_state("exp_entries", 1, {"title": "", "company": "", "duration": "", "bullets": ""})
    _init_list_state("proj_entries", 1, {"name": "", "description": "", "tech": ""})

    tab_personal, tab_edu, tab_exp, tab_proj, tab_extra = st.tabs(
        ["Personal & Objective", "Education", "Experience", "Projects", "Skills & Extras"]
    )

    with tab_personal:
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full Name*", key="b_full_name")
            email = st.text_input("Email*", key="b_email")
            phone = st.text_input("Phone", key="b_phone")
        with c2:
            location = st.text_input("Location", key="b_location")
            linkedin = st.text_input("LinkedIn URL", key="b_linkedin")
            github = st.text_input("GitHub URL", key="b_github")
        objective = st.text_area(
            "Career Objective / Summary",
            key="b_objective",
            placeholder="Results-driven Software Engineer with 3+ years of experience in...",
            height=100,
        )

    with tab_edu:
        st.caption("Add one or more education entries.")
        for i, entry in enumerate(st.session_state.edu_entries):
            with st.container(border=True):
                c1, c2 = st.columns(2)
                entry["degree"] = c1.text_input("Degree", value=entry["degree"], key=f"edu_degree_{i}")
                entry["school"] = c2.text_input("School / University", value=entry["school"], key=f"edu_school_{i}")
                c3, c4 = st.columns(2)
                entry["year"] = c3.text_input("Year", value=entry["year"], key=f"edu_year_{i}")
                entry["details"] = c4.text_input("Details (GPA, honors)", value=entry["details"], key=f"edu_details_{i}")
        bcol1, bcol2 = st.columns([1, 1])
        if bcol1.button("+ Add Education Entry"):
            st.session_state.edu_entries.append({"degree": "", "school": "", "year": "", "details": ""})
            st.rerun()
        if len(st.session_state.edu_entries) > 1 and bcol2.button("- Remove Last Education Entry"):
            st.session_state.edu_entries.pop()
            st.rerun()

    with tab_exp:
        st.caption("Add one or more work experience entries. Enter bullet points one per line.")
        for i, entry in enumerate(st.session_state.exp_entries):
            with st.container(border=True):
                c1, c2 = st.columns(2)
                entry["title"] = c1.text_input("Job Title", value=entry["title"], key=f"exp_title_{i}")
                entry["company"] = c2.text_input("Company", value=entry["company"], key=f"exp_company_{i}")
                entry["duration"] = st.text_input("Duration (e.g. Jan 2022 - Present)", value=entry["duration"], key=f"exp_duration_{i}")
                entry["bullets"] = st.text_area(
                    "Achievements / Responsibilities (one per line)",
                    value=entry["bullets"], key=f"exp_bullets_{i}", height=100,
                    placeholder="Developed X using Y, improving Z by 20%",
                )
        bcol1, bcol2 = st.columns([1, 1])
        if bcol1.button("+ Add Experience Entry"):
            st.session_state.exp_entries.append({"title": "", "company": "", "duration": "", "bullets": ""})
            st.rerun()
        if len(st.session_state.exp_entries) > 1 and bcol2.button("- Remove Last Experience Entry"):
            st.session_state.exp_entries.pop()
            st.rerun()

    with tab_proj:
        st.caption("Showcase key projects.")
        for i, entry in enumerate(st.session_state.proj_entries):
            with st.container(border=True):
                entry["name"] = st.text_input("Project Name", value=entry["name"], key=f"proj_name_{i}")
                entry["description"] = st.text_area("Description", value=entry["description"], key=f"proj_desc_{i}", height=70)
                entry["tech"] = st.text_input("Technologies Used", value=entry["tech"], key=f"proj_tech_{i}")
        bcol1, bcol2 = st.columns([1, 1])
        if bcol1.button("+ Add Project"):
            st.session_state.proj_entries.append({"name": "", "description": "", "tech": ""})
            st.rerun()
        if len(st.session_state.proj_entries) > 1 and bcol2.button("- Remove Last Project"):
            st.session_state.proj_entries.pop()
            st.rerun()

    with tab_extra:
        skills_raw = st.text_area("Skills (comma-separated)", key="b_skills",
                                   placeholder="Python, SQL, AWS, Docker")
        certs_raw = st.text_area("Certifications (one per line)", key="b_certs")
        achievements_raw = st.text_area("Achievements (one per line)", key="b_achievements")
        c1, c2 = st.columns(2)
        languages_raw = c1.text_input("Languages (comma-separated)", key="b_languages")
        interests_raw = c2.text_input("Interests (comma-separated)", key="b_interests")

    st.divider()
    st.markdown('<div class="section-label">Template & Generate</div>', unsafe_allow_html=True)
    template = st.radio("Choose a template", ["Modern", "Professional", "Minimal"], horizontal=True)

    generate_col, analyze_col = st.columns([1, 1])
    generate_clicked = generate_col.button("🚀 Generate Resume PDF", type="primary", width='stretch')

    if generate_clicked:
        if not st.session_state.get("b_full_name") or not st.session_state.get("b_email"):
            st.warning("Please fill in at least your Full Name and Email before generating.")
        else:
            data = ResumeData(
                full_name=st.session_state.b_full_name,
                email=st.session_state.b_email,
                phone=st.session_state.get("b_phone", ""),
                location=st.session_state.get("b_location", ""),
                linkedin=st.session_state.get("b_linkedin", ""),
                github=st.session_state.get("b_github", ""),
                objective=st.session_state.get("b_objective", ""),
                education=[e for e in st.session_state.edu_entries if e["degree"] or e["school"]],
                experience=[
                    {
                        "title": e["title"], "company": e["company"], "duration": e["duration"],
                        "bullets": [b.strip() for b in e["bullets"].split("\n") if b.strip()],
                    }
                    for e in st.session_state.exp_entries if e["title"] or e["company"]
                ],
                projects=[p for p in st.session_state.proj_entries if p["name"]],
                skills=[s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else [],
                certifications=[c.strip() for c in certs_raw.split("\n") if c.strip()] if certs_raw else [],
                achievements=[a.strip() for a in achievements_raw.split("\n") if a.strip()] if achievements_raw else [],
                languages=[l.strip() for l in languages_raw.split(",") if l.strip()] if languages_raw else [],
                interests=[i.strip() for i in interests_raw.split(",") if i.strip()] if interests_raw else [],
            )

            pdf_bytes = generate_resume_pdf(data, template=template)
            plain_text = resume_data_to_plain_text(data)

            st.session_state.resume_data = data
            st.session_state.last_resume_text = plain_text
            st.session_state.built_pdf_bytes = pdf_bytes

            try:
                save_resume_record(name=data.full_name, resume_text=plain_text, source="builder")
            except Exception:
                pass  # DB persistence is best-effort; don't block PDF generation on it

            st.success("Resume generated successfully!")

    if st.session_state.get("built_pdf_bytes"):
        st.download_button(
            "⬇️ Download Resume PDF",
            data=st.session_state.built_pdf_bytes,
            file_name=f"{st.session_state.get('b_full_name', 'resume').replace(' ', '_')}_resume.pdf",
            mime="application/pdf",
            width='stretch',
        )
        st.info("Head to **Resume Analyzer**, **ATS Score Predictor**, or **Job Matching** in the "
                "sidebar — your generated resume text is now available there automatically.")
