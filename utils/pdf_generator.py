"""
utils/pdf_generator.py
------------------------
Generates ATS-friendly PDF resumes from structured form data using
ReportLab. Three visual templates are supported: Modern, Professional,
and Minimal. All templates use single-column, text-selectable layouts
(no tables-as-images, no icons-as-text) to remain ATS-parseable.
"""

import io
from dataclasses import dataclass, field
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


@dataclass
class ResumeData:
    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    objective: str = ""
    education: List[dict] = field(default_factory=list)     # {degree, school, year, details}
    experience: List[dict] = field(default_factory=list)    # {title, company, duration, bullets: []}
    projects: List[dict] = field(default_factory=list)      # {name, description, tech}
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)


TEMPLATE_COLORS = {
    "Modern": {"accent": colors.HexColor("#2563EB"), "text": colors.HexColor("#1F2937"),
               "muted": colors.HexColor("#6B7280")},
    "Professional": {"accent": colors.HexColor("#1F2937"), "text": colors.HexColor("#111827"),
                      "muted": colors.HexColor("#4B5563")},
    "Minimal": {"accent": colors.HexColor("#000000"), "text": colors.HexColor("#000000"),
                "muted": colors.HexColor("#555555")},
}


def _get_styles(template: str):
    palette = TEMPLATE_COLORS.get(template, TEMPLATE_COLORS["Modern"])
    font_main = "Helvetica"
    font_bold = "Helvetica-Bold"

    name_style = ParagraphStyle(
        "Name", fontName=font_bold, fontSize=22 if template != "Minimal" else 18,
        textColor=palette["text"], spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        "Contact", fontName=font_main, fontSize=9.5, textColor=palette["muted"], spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section", fontName=font_bold, fontSize=12, textColor=palette["accent"],
        spaceBefore=12, spaceAfter=4, letterSpacing=0.5,
    )
    body_style = ParagraphStyle(
        "Body", fontName=font_main, fontSize=10, textColor=palette["text"],
        leading=14, spaceAfter=2,
    )
    subheading_style = ParagraphStyle(
        "Subheading", fontName=font_bold, fontSize=10.5, textColor=palette["text"], spaceAfter=1,
    )
    meta_style = ParagraphStyle(
        "Meta", fontName=font_main, fontSize=9, textColor=palette["muted"],
        spaceAfter=4, italic=True,
    )
    bullet_style = ParagraphStyle(
        "Bullet", fontName=font_main, fontSize=10, textColor=palette["text"], leading=13,
    )
    return {
        "name": name_style, "contact": contact_style, "section": section_style,
        "body": body_style, "subheading": subheading_style, "meta": meta_style,
        "bullet": bullet_style, "palette": palette,
    }


def _section_header(text: str, styles) -> Paragraph:
    return Paragraph(text.upper(), styles["section"])


def _divider(styles, template: str):
    thickness = 1.4 if template == "Professional" else 0.75
    return HRFlowable(width="100%", thickness=thickness, color=styles["palette"]["accent"],
                       spaceBefore=0, spaceAfter=8)


def generate_resume_pdf(data: ResumeData, template: str = "Modern") -> bytes:
    """Builds a PDF resume in-memory and returns the raw bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = _get_styles(template)
    story = []

    # Header
    story.append(Paragraph(data.full_name or "Your Name", styles["name"]))
    contact_bits = [b for b in [data.email, data.phone, data.location, data.linkedin, data.github] if b]
    story.append(Paragraph(" | ".join(contact_bits), styles["contact"]))
    story.append(_divider(styles, template))

    # Objective / Summary
    if data.objective:
        story.append(_section_header("Career Objective", styles))
        story.append(Paragraph(data.objective, styles["body"]))

    # Experience
    if data.experience:
        story.append(_section_header("Experience", styles))
        for job in data.experience:
            title_line = f'{job.get("title", "")} — {job.get("company", "")}'
            story.append(Paragraph(title_line, styles["subheading"]))
            if job.get("duration"):
                story.append(Paragraph(job["duration"], styles["meta"]))
            bullets = [b for b in job.get("bullets", []) if b.strip()]
            for b in bullets:
                story.append(Paragraph(f"- {b}", styles["bullet"]))
            story.append(Spacer(1, 6))

    # Projects
    if data.projects:
        story.append(_section_header("Projects", styles))
        for proj in data.projects:
            story.append(Paragraph(proj.get("name", ""), styles["subheading"]))
            if proj.get("description"):
                story.append(Paragraph(proj["description"], styles["body"]))
            if proj.get("tech"):
                story.append(Paragraph(f'Technologies: {proj["tech"]}', styles["meta"]))
            story.append(Spacer(1, 6))

    # Education
    if data.education:
        story.append(_section_header("Education", styles))
        for edu in data.education:
            line = f'{edu.get("degree", "")} — {edu.get("school", "")}'
            story.append(Paragraph(line, styles["subheading"]))
            meta_bits = [edu.get("year", ""), edu.get("details", "")]
            meta_line = " | ".join([m for m in meta_bits if m])
            if meta_line:
                story.append(Paragraph(meta_line, styles["meta"]))
            story.append(Spacer(1, 4))

    # Skills
    if data.skills:
        story.append(_section_header("Skills", styles))
        story.append(Paragraph(", ".join(data.skills), styles["body"]))

    # Certifications
    if data.certifications:
        story.append(_section_header("Certifications", styles))
        for c in data.certifications:
            story.append(Paragraph(f"- {c}", styles["bullet"]))

    # Achievements
    if data.achievements:
        story.append(_section_header("Achievements", styles))
        for a in data.achievements:
            story.append(Paragraph(f"- {a}", styles["bullet"]))

    # Languages
    if data.languages:
        story.append(_section_header("Languages", styles))
        story.append(Paragraph(", ".join(data.languages), styles["body"]))

    # Interests
    if data.interests:
        story.append(_section_header("Interests", styles))
        story.append(Paragraph(", ".join(data.interests), styles["body"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def resume_data_to_plain_text(data: ResumeData) -> str:
    """Flattens ResumeData into plain text for feeding into the ML models
    (classifier / ATS scorer / job matcher) after building via the form UI."""
    parts = [data.full_name, data.objective]
    for edu in data.education:
        parts.append(f'{edu.get("degree", "")} {edu.get("school", "")} {edu.get("details", "")}')
    for job in data.experience:
        parts.append(f'{job.get("title", "")} {job.get("company", "")} {job.get("duration", "")}')
        parts.extend(job.get("bullets", []))
    for proj in data.projects:
        parts.append(f'{proj.get("name", "")} {proj.get("description", "")} {proj.get("tech", "")}')
    parts.append(", ".join(data.skills))
    parts.append(", ".join(data.certifications))
    parts.append(", ".join(data.achievements))
    parts.append(", ".join(data.languages))
    parts.append(", ".join(data.interests))
    return "\n".join(p for p in parts if p)
