import os
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# Google Gemini API Setup
# ==============================
API_KEY = st.secrets.get("API_KEY") or os.environ.get("API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ==============================
# Loading  Models
# ==============================
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
    tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_roberta_model():
    return pipeline("zero-shot-classification", model="roberta-large-mnli")

bert_pipeline = None
roberta_pipeline = None

def get_bert_model():
    global bert_pipeline
    if bert_pipeline is None:
        with st.spinner("Loading BERT model..."):
            bert_pipeline = load_bert_model()
    return bert_pipeline

def get_roberta_model():
    global roberta_pipeline
    if roberta_pipeline is None:
        with st.spinner("Loading RoBERTa model..."):
            roberta_pipeline = load_roberta_model()
    return roberta_pipeline

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
            chunks = [elem.get_text().strip() for elem in article_div.find_all(["p", "li", "div"]) if len(elem.get_text().split()) > 5]
        else:
            chunks = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split()) > 5]
        text = " ".join(chunks)
        if not text:
            text = soup.get_text()
        return clean_text((title + "\n\n" + text)[:4000])
    except Exception:
        return None

# ==============================
# Trusted Sources
# ==============================
trusted_sources = [
    "thehindu.com","timesofindia.com","hindustantimes.com","ndtv.com","indiatoday.in", "indianexpress.com","livemint.com","business-standard.com","deccanherald.com", "telegraphindia.com","mid-day.com","dnaindia.com","scroll.in","firstpost.com", "theprint.in","news18.com","oneindia.com","outlookindia.com","zeenews.india.com", "cnnnews18.com","economictimes.indiatimes.com","financialexpress.com","siasat.com", "newindianexpress.com","tribuneindia.com","asianage.com","bharattimes.com", "freepressjournal.in","morningindia.in","abplive.com","newsable.asianetnews.com","bbc.com","cnn.com","reuters.com","apnews.com","aljazeera.com","theguardian.com", "nytimes.com","washingtonpost.com","bloomberg.com","dw.com","foxnews.com","cbsnews.com", "nbcnews.com","abcnews.go.com","sky.com","france24.com","rt.com","sputniknews.com", "npr.org","telegraph.co.uk","thetimes.co.uk","independent.co.uk","globaltimes.cn", "china.org.cn","cbc.ca","abc.net.au","smh.com.au","japantimes.co.jp","lemonde.fr", "elpais.com","derstandard.at","spiegel.de","tagesschau.de","asiatimes.com", "straitstimes.com","thaiworldview.com","thejakartapost.com","thestandard.com.hk", "sbs.com.au","hawaiinewsnow.com","theglobeandmail.com","irishnews.com","latimes.com", "chicagotribune.com","startribune.com","nydailynews.com","financialtimes.com", "forbes.com","thehill.com","vox.com","buzzfeednews.com","huffpost.com","usatoday.com", "teleSURenglish.net","euronews.com","al-monitor.com","news.com.au","cnbc.com", "barrons.com","time.com","foreignpolicy.com","economist.com","foreignaffairs.com", "dailytelegraph.com.au","smh.com.au","thesun.co.uk","dailymail.co.uk",".gov.in","pib.gov.in","isro.gov.in","pmindia.gov.in","mod.gov.in","mha.gov.in", "rbi.org.in","sebi.gov.in","nic.in","mohfw.gov.in","moef.gov.in","meity.gov.in", "railway.gov.in","dgca.gov.in","drdo.gov.in","indianrailways.gov.in","education.gov.in", "scienceandtech.gov.in","urbanindia.nic.in","financialservices.gov.in", "commerce.gov.in","sportsauthorityofindia.nic.in","agriculture.gov.in","power.gov.in", "parliamentofindia.nic.in","taxindia.gov.in","cbic.gov.in","epfindia.gov.in","defence.gov.in",".gov",".europa.eu","un.org","who.int","nasa.gov","esa.int","imf.org","worldbank.org", "fao.org","wto.org","unicef.org","unhcr.org","redcross.org","cdc.gov","nih.gov","usa.gov", "canada.ca","gov.uk","australia.gov.au","japan.go.jp","ec.europa.eu","consilium.europa.eu", "ecb.europa.eu","unep.org","ilo.org","ohchr.org","unodc.org","unwomen.org", "unfpa.org","unesco.org","wmo.int","ifrc.org","nato.int","oecd.org","europarl.europa.eu", "unido.org","wfp.org"
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

    bert_model = get_bert_model()
    roberta_model = get_roberta_model()

    bert_res = bert_model(text[:512])[0]['label']
    bert_pred = "REAL" if "REAL" in bert_res.upper() else "FAKE"

    roberta_res = roberta_model(text, candidate_labels=["REAL", "FAKE"])
    roberta_pred = roberta_res['labels'][0]

    scores = {"REAL": 0, "FAKE": 0}
    for p, w in zip([bert_pred, roberta_pred], [0.5, 0.5]):
        if p == "REAL":
            scores["REAL"] += w
        elif p == "FAKE":
            scores["FAKE"] += w

    return "REAL" if scores["REAL"] > scores["FAKE"] else "FAKE"

# ==============================
# Gemini API Query
# ==============================
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
    except Exception:
        return "UNSURE", "Explanation not available."

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection App (DL + Gemini)")

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
            with st.spinner("Analyzing using deep learning models..."):
                final_result = predict_text_ensemble(user_input, page_url)
            with st.spinner("Getting Gemini AI verification..."):
                gemini_label, gemini_explanation = query_api(user_input)

            st.subheader("Final Verdicts")
            if final_result == "REAL":
                st.success(f"🟢 DL Ensemble: REAL NEWS")
            else:
                st.error(f"🔴 DL Ensemble: FAKE NEWS")

            st.info(f" Verdict: {gemini_label}")
            st.write(f"💬 {gemini_explanation}")

        except Exception as e:
            st.error(f"⚠️ Error during analysis: {e}")
