"""
utils/text_processing.py
-------------------------
Shared NLP text-cleaning utilities used by both train.py (offline model
training) and the Streamlit app (live inference). Keeping this logic in
one place guarantees the exact same preprocessing is applied at train
and inference time -- a common source of bugs when it's duplicated.
"""

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# spaCy is used for noun-phrase extraction (smarter "recommended keyword"
# suggestions in job matching, beyond the fixed skills vocabulary below).
# It's optional at runtime: if the model isn't installed the app still
# works using only the regex/NLTK pipeline -- see extract_key_phrases().
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        _NLP = None
except ImportError:
    _NLP = None

# Ensure required NLTK corpora are available. Downloads are silent/no-op
# if already present (Streamlit Cloud runs this once on cold start).
for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(
            f"corpora/{pkg}" if pkg not in ("punkt", "punkt_tab") else f"tokenizers/{pkg}"
        )
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/phone numbers/punctuation/digits."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> List[str]:
    """Tokenize, remove stopwords/short tokens, and lemmatize."""
    tokens = word_tokenize(text)
    cleaned = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 2 and tok not in string.punctuation
    ]
    return cleaned


def preprocess_for_model(text: str) -> str:
    """
    Full pipeline used before TF-IDF vectorization:
    clean -> tokenize -> remove stopwords -> lemmatize -> rejoin.
    This is the single function both train.py and app.py should call.
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


def extract_years_of_experience(text: str) -> int:
    """Best-effort regex extraction of years of experience from resume text."""
    match = re.search(r"(\d+)\+?\s*years?", text.lower())
    if match:
        return int(match.group(1))
    return 0


COMMON_SKILLS = [
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "r",
    "html", "css", "react", "angular", "vue", "node.js", "express.js",
    "django", "flask", "fastapi", "spring boot", "hibernate",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "machine learning", "deep learning", "nlp", "computer vision", "mlops",
    "docker", "kubernetes", "jenkins", "terraform", "ansible", "ci/cd",
    "aws", "azure", "google cloud platform", "gcp", "linux", "bash scripting",
    "git", "rest apis", "microservices", "mongodb", "mysql", "postgresql",
    "redis", "kafka", "tableau", "power bi", "excel", "statistics",
    "data structures", "algorithms", "system design", "agile", "scrum",
    "junit", "maven", "multithreading", "design patterns", "oop",
]


def extract_skills(text: str) -> List[str]:
    """Match against a known skills vocabulary (case-insensitive substring match)."""
    text_lower = text.lower()
    found = []
    for skill in COMMON_SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def spacy_available() -> bool:
    return _NLP is not None


def extract_key_phrases(text: str, top_n: int = 15) -> List[str]:
    """
    Uses spaCy noun-chunk extraction to surface multi-word technical/role
    terms (e.g. "cloud infrastructure", "machine learning models") that
    the fixed COMMON_SKILLS vocabulary wouldn't catch. Falls back to an
    empty list if the spaCy model isn't installed, so callers should treat
    this as a nice-to-have enhancement, not a required signal.
    """
    if _NLP is None or not text.strip():
        return []

    doc = _NLP(text[:20000])  # cap length for performance on long resumes
    phrases = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        phrase = re.sub(r"^(the|a|an|our|your|my|this|these|those)\s+", "", phrase)
        words = phrase.split()
        if 1 < len(words) <= 4 and all(w.isalpha() or w in {"-", "/"} for w in words):
            phrases.append(phrase)

    # Preserve order of first appearance, dedupe, cap to top_n
    seen = []
    for p in phrases:
        if p not in seen:
            seen.append(p)
        if len(seen) >= top_n:
            break
    return seen
