import os
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# Load DL Models
# ==============================
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
    tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_roberta_model():
    return pipeline("zero-shot-classification", model="roberta-large-mnli")

bert_pipeline = load_bert_model()
roberta_pipeline = load_roberta_model()

# ==============================
# Text Cleaning
# ==============================
def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|days|weeks|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more|subscribe|follow us)", "", text, flags=re.I)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^\w\s,.!?;:()'\"]+", "", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\b\d{5,}\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# ==============================
# Web Scraping
# ==============================
def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        article_div = soup.find("article") or soup.find("div", {"class": "articlebodycontent"}) or soup.find("div", {"id": "content-body"})
        if article_div:
            chunks = [elem.get_text().strip() for elem in article_div.find_all(["p","li","div"]) if len(elem.get_text().split())>5]
        else:
            chunks = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split())>5]
        text = " ".join(chunks)
        if not text:
            text = soup.get_text()
        return clean_text((title + "\n\n" + text)[:4000])
    except:
        return None

# ==============================
# Trusted Sources
# ==============================
trusted_sources = [
    # Indian and International as before...
]

def is_trusted(url):
    url = url.lower()
    return any(src in url for src in trusted_sources)

# ==============================
# DL Ensemble Prediction
# ==============================
def predict_text_ensemble(text, url=""):
    text = clean_text(text)

    if url and is_trusted(url):
        return "REAL"

    bert_res = bert_pipeline(text[:512])[0]['label']
    bert_pred = "REAL" if "REAL" in bert_res.upper() else "FAKE"

    roberta_res = roberta_pipeline(text, candidate_labels=["REAL","FAKE"])
    roberta_pred = roberta_res['labels'][0]

    scores = {"REAL":0, "FAKE":0}
    for p, w in zip([bert_pred, roberta_pred],[0.5,0.5]):
        if p=="REAL": scores["REAL"] += w
        elif p=="FAKE": scores["FAKE"] += w

    return "REAL" if scores["REAL"] > scores["FAKE"] else "FAKE"

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection App (DL Ensemble)")

input_type = st.radio("Choose Input Type", ["Text", "URL"])

user_input = ""
page_url = ""

if input_type == "Text":
    user_input = st.text_area("Enter news text here", height=200)
elif input_type == "URL":
    page_url = st.text_input("Enter news article URL")
    if page_url:
        scraped = scrape_url(page_url)
        if scraped:
            st.text_area("Extracted Article", scraped, height=300)
            user_input = scraped
        else:
            st.warning("⚠️ Could not scrape the URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter valid text or URL.")
    else:
        try:
            final_result = predict_text_ensemble(user_input, page_url)
            st.subheader("Final Verdict:")
            if final_result=="REAL":
                st.success("🟢 REAL NEWS")
            elif final_result=="FAKE":
                st.error("🔴 FAKE NEWS")

        except Exception:
            st.error("⚠️ Error during analysis. Please try again.")

# ==============================
# Hidden Remote API Query
# ==============================
API_KEY = st.secrets.get("API_KEY") or os.environ.get("API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def query_api(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": f"Decide authenticity and provide a short explanation:\n{text}"}]}
        ]
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        raw_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        lines = raw_text.split("\n", 1)
        classification = lines[0].strip().upper() if lines else "UNSURE"
        explanation = lines[1].strip() if len(lines) > 1 else "No explanation."

        if "REAL" in classification:
            return "REAL", explanation
        elif "FAKE" in classification:
            return "FAKE", explanation
        else:
            return "UNSURE", explanation
    except:
        return "UNSURE", "Explanation not available."
