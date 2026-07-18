"""
utils/skill_recommender.py
----------------------------
Recommends additional skills to add to a resume based on the predicted
job category and the skills already detected in the resume text.
"""

from typing import Dict, List

from utils.text_processing import extract_skills

# Curated "in-demand" skill sets per category, ordered by typical priority.
# These mirror the pools used in dataset generation so recommendations feel
# consistent with what the classifier was trained to recognize.
RECOMMENDED_SKILLS_BY_CATEGORY: Dict[str, List[str]] = {
    "Data Scientist": ["tensorflow", "pytorch", "power bi", "tableau", "statistics",
                        "deep learning", "aws", "docker"],
    "AI Engineer": ["docker", "kubernetes", "aws", "mlops", "fastapi",
                     "computer vision", "nlp", "git"],
    "Software Engineer": ["system design", "microservices", "docker", "ci/cd",
                           "design patterns", "rest apis", "agile"],
    "Web Developer": ["typescript", "next.js", "redux", "docker", "rest apis",
                       "git", "mongodb"],
    "Java Developer": ["spring boot", "kafka", "docker", "kubernetes",
                        "microservices", "junit", "aws"],
    "DevOps Engineer": ["terraform", "kubernetes", "aws", "prometheus",
                         "ansible", "ci/cd", "azure"],
    "Cloud Engineer": ["kubernetes", "terraform", "azure", "google cloud platform",
                        "security", "iam", "python"],
}


def recommend_skills(resume_text: str, predicted_category: str, top_n: int = 8) -> Dict:
    detected = extract_skills(resume_text)
    detected_set = {s.lower() for s in detected}

    pool = RECOMMENDED_SKILLS_BY_CATEGORY.get(predicted_category, [])
    recommended = [s for s in pool if s.lower() not in detected_set][:top_n]

    return {
        "detected_skills": detected,
        "recommended_skills": recommended,
        "category": predicted_category,
    }
