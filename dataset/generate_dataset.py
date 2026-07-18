"""
generate_dataset.py
--------------------
Generates a realistic synthetic resume dataset for training the ML models
used in the AI Resume Builder project.

Why synthetic data?
Public resume datasets (e.g. Kaggle's "Resume Dataset") are not reachable
from this build environment (no internet access to Kaggle), so this script
programmatically builds a dataset with the SAME structure and statistical
properties (skills, job categories, experience levels, ATS-relevant text
patterns) that those datasets have. Swap this file's output for a real
Kaggle CSV at any time -- train.py only expects columns:
['Resume', 'Category'] for classification and
['Resume', 'JobDescription', 'ATS_Score'] for the ATS regressor.

Run:
    python dataset/generate_dataset.py
Outputs:
    dataset/resume_dataset.csv          (classification)
    dataset/ats_dataset.csv             (ATS score regression)
    dataset/job_descriptions.csv        (job matching)
"""

import random
import csv
import os

random.seed(42)

CATEGORIES = [
    "Data Scientist",
    "AI Engineer",
    "Software Engineer",
    "Web Developer",
    "Java Developer",
    "DevOps Engineer",
    "Cloud Engineer",
]

# Skill pools per category (also reused by the live app's skill recommender)
SKILLS_BY_CATEGORY = {
    "Data Scientist": ["Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn",
                        "TensorFlow", "PyTorch", "Statistics", "Data Visualization",
                        "Tableau", "Power BI", "Machine Learning", "Deep Learning",
                        "Feature Engineering", "A/B Testing", "Jupyter", "Matplotlib"],
    "AI Engineer": ["Python", "TensorFlow", "PyTorch", "Keras", "NLP",
                     "Computer Vision", "MLOps", "Docker", "Kubernetes",
                     "Transformers", "Hugging Face", "OpenCV", "CUDA",
                     "Model Deployment", "Flask", "FastAPI", "REST APIs"],
    "Software Engineer": ["Java", "C++", "Python", "Data Structures", "Algorithms",
                           "Git", "System Design", "Object Oriented Programming",
                           "REST APIs", "Unit Testing", "Agile", "SQL", "Design Patterns",
                           "Microservices", "CI/CD"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express.js",
                       "MongoDB", "REST APIs", "TypeScript", "Redux", "Next.js",
                       "Bootstrap", "Tailwind CSS", "Webpack", "Git", "SQL"],
    "Java Developer": ["Java", "Spring Boot", "Hibernate", "Maven", "REST APIs",
                        "Microservices", "SQL", "MySQL", "JUnit", "Design Patterns",
                        "Multithreading", "Git", "Docker", "Kafka", "JPA"],
    "DevOps Engineer": ["Docker", "Kubernetes", "Jenkins", "AWS", "Terraform",
                         "Ansible", "CI/CD", "Linux", "Bash Scripting", "Git",
                         "Prometheus", "Grafana", "Azure DevOps", "GitLab CI",
                         "Monitoring", "Infrastructure as Code"],
    "Cloud Engineer": ["AWS", "Azure", "Google Cloud Platform", "Terraform",
                        "Kubernetes", "Docker", "CloudFormation", "Lambda",
                        "S3", "EC2", "Networking", "Security", "IAM",
                        "DevOps", "Linux", "Python"],
}

EXTRA_SKILLS = ["Communication", "Team Leadership", "Problem Solving",
                "Project Management", "Time Management", "Agile", "Scrum"]

DEGREES = ["B.Tech in Computer Science", "B.E in Information Technology",
           "M.Tech in Computer Science", "MCA", "B.Sc in Computer Science",
           "M.Sc in Data Science", "B.Tech in Electronics and Communication"]

UNIVERSITIES = ["Indian Institute of Technology", "National Institute of Technology",
                "Anna University", "Pune University", "VIT University",
                "State Technical University", "Delhi Technological University"]

COMPANIES = ["TechCorp Solutions", "Innovate Systems", "DataWave Analytics",
             "CloudNine Technologies", "NexGen Software", "ByteForge Inc",
             "Quantum Softworks", "Pinnacle IT Services", "Skyline Digital",
             "Vertex Solutions"]

ACTION_VERBS = ["Developed", "Designed", "Implemented", "Led", "Optimized",
                "Built", "Architected", "Automated", "Deployed", "Managed",
                "Improved", "Engineered", "Created", "Streamlined", "Delivered"]

OBJECTS = {
    "Data Scientist": ["predictive models", "data pipelines", "machine learning models",
                        "statistical analyses", "dashboards", "recommendation systems"],
    "AI Engineer": ["deep learning models", "NLP pipelines", "computer vision systems",
                     "AI inference services", "model training pipelines", "chatbots"],
    "Software Engineer": ["backend services", "APIs", "software modules",
                           "scalable systems", "automated tests", "internal tools"],
    "Web Developer": ["web applications", "responsive UIs", "frontend components",
                       "e-commerce platforms", "single page applications", "landing pages"],
    "Java Developer": ["Spring Boot microservices", "REST APIs", "backend systems",
                        "enterprise applications", "database integrations", "batch jobs"],
    "DevOps Engineer": ["CI/CD pipelines", "deployment automation", "monitoring systems",
                         "infrastructure as code", "container orchestration", "release pipelines"],
    "Cloud Engineer": ["cloud infrastructure", "serverless architectures", "cloud migrations",
                        "auto-scaling systems", "cloud security policies", "multi-region deployments"],
}

CERTIFICATIONS = ["AWS Certified Solutions Architect", "Google Data Analytics Certificate",
                   "Microsoft Certified: Azure Fundamentals", "Certified Kubernetes Administrator",
                   "Oracle Certified Java Programmer", "TensorFlow Developer Certificate",
                   "PMP Certification", "Scrum Master Certification"]


def random_experience_bullets(category, n=3):
    bullets = []
    for _ in range(n):
        verb = random.choice(ACTION_VERBS)
        obj = random.choice(OBJECTS[category])
        skill = random.choice(SKILLS_BY_CATEGORY[category])
        metric = random.choice(["by 20%", "by 35%", "resulting in 15% cost savings",
                                 "improving performance by 40%", "reducing latency by 25%",
                                 "for over 10,000 users", "across 5 production environments"])
        bullets.append(f"{verb} {obj} using {skill}, {metric}.")
    return bullets


def generate_resume_text(category, experience_years):
    skills = random.sample(SKILLS_BY_CATEGORY[category],
                            k=min(8, len(SKILLS_BY_CATEGORY[category])))
    skills += random.sample(EXTRA_SKILLS, k=2)
    degree = random.choice(DEGREES)
    university = random.choice(UNIVERSITIES)
    company = random.choice(COMPANIES)
    bullets = random_experience_bullets(category, n=random.randint(2, 4))
    cert = random.choice(CERTIFICATIONS)

    resume = f"""
Professional Summary:
Results-driven {category} with {experience_years} years of experience in
{', '.join(skills[:4])}. Proven track record of delivering high quality
solutions in fast paced environments.

Skills:
{', '.join(skills)}

Experience:
{category} at {company}
{chr(10).join('- ' + b for b in bullets)}

Education:
{degree}, {university}

Certifications:
{cert}
"""
    return resume.strip(), skills


def generate_job_description(category):
    skills = random.sample(SKILLS_BY_CATEGORY[category],
                            k=min(6, len(SKILLS_BY_CATEGORY[category])))
    years = random.choice([1, 2, 3, 4, 5])
    jd = f"""
We are looking for a {category} with at least {years} years of experience.
Required Skills: {', '.join(skills)}.
Responsibilities include working on {', '.join(random.sample(OBJECTS[category], k=2))}
and collaborating with cross-functional teams. Strong communication and
problem-solving skills required.
"""
    return jd.strip(), skills


def compute_synthetic_ats_score(resume_text, skills, experience_years):
    """
    Rough heuristic used ONLY to label synthetic training data.
    Real signal in the app comes from the trained regression model,
    not this function -- this just gives the model something
    plausible to learn from.
    """
    score = 40
    score += min(len(skills), 10) * 2          # skill coverage
    score += min(experience_years, 10) * 1.5   # experience
    score += 5 if "Certifications" in resume_text else 0
    length_bonus = min(len(resume_text.split()) / 20, 10)
    score += length_bonus
    score += random.uniform(-8, 8)             # noise
    return max(0, min(100, round(score, 1)))


def build_datasets(n_per_category=120):
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    resume_rows = []
    ats_rows = []
    jd_rows = []

    for category in CATEGORIES:
        for _ in range(n_per_category):
            years = random.randint(0, 12)
            resume_text, skills = generate_resume_text(category, years)
            resume_rows.append({"Resume": resume_text, "Category": category})

            jd_text, jd_skills = generate_job_description(category)
            ats_score = compute_synthetic_ats_score(resume_text, skills, years)
            ats_rows.append({
                "Resume": resume_text,
                "JobDescription": jd_text,
                "ExperienceYears": years,
                "ATS_Score": ats_score,
            })

        # a handful of standalone job descriptions per category for matching demos
        for _ in range(5):
            jd_text, jd_skills = generate_job_description(category)
            jd_rows.append({"Category": category, "JobDescription": jd_text})

    random.shuffle(resume_rows)
    random.shuffle(ats_rows)

    out_dir = os.path.dirname(__file__)

    with open(os.path.join(out_dir, "resume_dataset.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Resume", "Category"])
        writer.writeheader()
        writer.writerows(resume_rows)

    with open(os.path.join(out_dir, "ats_dataset.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Resume", "JobDescription", "ExperienceYears", "ATS_Score"])
        writer.writeheader()
        writer.writerows(ats_rows)

    with open(os.path.join(out_dir, "job_descriptions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Category", "JobDescription"])
        writer.writeheader()
        writer.writerows(jd_rows)

    print(f"Generated {len(resume_rows)} resumes across {len(CATEGORIES)} categories.")
    print(f"Generated {len(ats_rows)} ATS-labeled samples.")
    print(f"Generated {len(jd_rows)} sample job descriptions.")


if __name__ == "__main__":
    build_datasets(n_per_category=120)
