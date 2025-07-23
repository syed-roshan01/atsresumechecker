from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

def extract_resume_text(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_keywords(text):
    doc = nlp(text)
    return set(token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"])

def check_ats_compatibility(resume_text):
    issues = []
    if re.search(r"\[.*?\]", resume_text):
        issues.append("Avoid square brackets (e.g., [Your Name]).")
    if re.search(r"\bMgr\b|\bSr\b", resume_text, re.IGNORECASE):
        issues.append("Use full words like Manager or Senior instead of abbreviations.")
    if len(resume_text.split()) > 1000:
        issues.append("Resume too long; aim for 1-2 pages.")
    return issues

def generate_feedback(resume_keywords, job_keywords, match_score):
    feedback = []
    missing_keywords = job_keywords - resume_keywords
    if missing_keywords:
        feedback.append(f"Add these keywords: {', '.join(missing_keywords)}")
    if match_score < 0.75:
        feedback.append("Match score below 75%. Tailor skills to job description.")
    feedback.append("Use standard fonts (e.g., Arial) and single-column layout.")
    return feedback

@app.post("/analyze_resume")
async def analyze_resume(resume: UploadFile = File(...), job_desc: str = Form(...)):
    resume_text = extract_resume_text(resume.file)
    if "Error" in resume_text:
        return JSONResponse({"error": resume_text}, status_code=400)
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_desc)
    vectorizer = TfidfVectorizer()
    documents = [resume_text, job_desc]
    tfidf_matrix = vectorizer.fit_transform(documents)
    match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    ats_issues = check_ats_compatibility(resume_text)
    feedback = generate_feedback(resume_keywords, job_keywords, match_score)
    return JSONResponse({
        "match_score": f"{match_score:.2%}",
        "ats_issues": ats_issues,
        "feedback": feedback,
        "resume_keywords": list(resume_keywords),
        "job_keywords": list(job_keywords)
    })
