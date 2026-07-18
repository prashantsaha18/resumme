"""
utils/ats_analysis.py
----------------------
Rule-based analysis layered on top of the trained ATS regression model.
The ML model produces the numeric score; this module turns that score
(plus simple structural checks on the resume text) into human-readable
strengths, weaknesses, and improvement suggestions for the UI.
"""

import re
from typing import Dict, List

from utils.text_processing import extract_skills, extract_years_of_experience

SECTION_KEYWORDS = {
    "summary": ["summary", "objective", "profile"],
    "experience": ["experience", "employment", "work history"],
    "education": ["education", "academic"],
    "skills": ["skills", "technical skills", "competencies"],
    "projects": ["projects"],
    "certifications": ["certification", "certificate"],
    "contact": ["email", "phone", "@"],
}

ACTION_VERBS = {
    "developed", "designed", "implemented", "led", "optimized", "built",
    "architected", "automated", "deployed", "managed", "improved",
    "engineered", "created", "streamlined", "delivered", "achieved",
    "increased", "reduced", "launched", "collaborated",
}


def check_sections(text: str) -> Dict[str, bool]:
    text_lower = text.lower()
    return {
        section: any(kw in text_lower for kw in keywords)
        for section, keywords in SECTION_KEYWORDS.items()
    }


def count_action_verbs(text: str) -> int:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return sum(1 for w in words if w in ACTION_VERBS)


def count_quantified_bullets(text: str) -> int:
    """Counts lines/bullets that contain a number (quantified impact)."""
    lines = text.split("\n")
    return sum(1 for line in lines if re.search(r"\d", line) and len(line.strip()) > 5)


def word_count(text: str) -> int:
    return len(text.split())


def analyze_resume(text: str, predicted_score: float) -> Dict:
    """
    Combines the ML-predicted ATS score with rule-based structural checks
    to produce strengths, weaknesses, and suggestions.
    """
    sections = check_sections(text)
    skills = extract_skills(text)
    action_verb_count = count_action_verbs(text)
    quantified = count_quantified_bullets(text)
    wc = word_count(text)
    years = extract_years_of_experience(text)

    strengths: List[str] = []
    weaknesses: List[str] = []
    suggestions: List[str] = []

    # Sections
    missing_sections = [s for s, present in sections.items() if not present]
    if not missing_sections:
        strengths.append("Resume includes all key sections (summary, experience, education, skills).")
    else:
        for s in missing_sections:
            weaknesses.append(f"Missing or unclear '{s.title()}' section.")
            suggestions.append(f"Add a clearly labeled '{s.title()}' section.")

    # Skills
    if len(skills) >= 8:
        strengths.append(f"Strong skills coverage detected ({len(skills)} relevant skills found).")
    elif len(skills) >= 4:
        strengths.append(f"Moderate skills coverage ({len(skills)} relevant skills found).")
        suggestions.append("Add a few more role-relevant technical skills to strengthen keyword matching.")
    else:
        weaknesses.append("Very few recognizable technical skills found.")
        suggestions.append("List specific tools, languages, and frameworks you've used (e.g., Python, AWS, React).")

    # Action verbs
    if action_verb_count >= 5:
        strengths.append("Good use of strong action verbs (e.g., Developed, Led, Optimized).")
    else:
        weaknesses.append("Limited use of strong action verbs in experience bullet points.")
        suggestions.append("Start bullet points with action verbs like 'Developed', 'Led', or 'Implemented'.")

    # Quantified achievements
    if quantified >= 3:
        strengths.append("Multiple quantified achievements found (numbers/metrics in bullet points).")
    else:
        weaknesses.append("Few quantified achievements (numbers, percentages, metrics).")
        suggestions.append("Quantify impact where possible, e.g. 'Improved performance by 30%'.")

    # Length
    if wc < 150:
        weaknesses.append("Resume content seems too short for ATS parsing and recruiter review.")
        suggestions.append("Expand on your experience and projects with more detail (aim for 300-700 words).")
    elif wc > 1000:
        weaknesses.append("Resume is quite long, which may dilute keyword density.")
        suggestions.append("Trim to the most relevant, recent, and impactful experience (1-2 pages).")
    else:
        strengths.append("Resume length is appropriate for ATS parsing.")

    # Experience
    if years == 0:
        suggestions.append("Explicitly state your total years of experience near the top of the resume.")

    if not strengths:
        strengths.append("Resume was successfully parsed and analyzed.")
    if not suggestions:
        suggestions.append("Resume is well optimized. Consider tailoring keywords per job description.")

    return {
        "score": round(float(predicted_score), 1),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "skills_found": skills,
        "sections": sections,
        "word_count": wc,
        "action_verb_count": action_verb_count,
        "quantified_bullets": quantified,
        "years_experience": years,
    }
