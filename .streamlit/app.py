# fake_news_app.py
import os
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Transformer imports are optional — we attempt to import them, but handle failures gracefully.
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    TRANSFORMERS_AVAILABLE = False

# ==============================
# Config / Secrets
# ==============================
API_KEY = st.secrets.get("API_KEY") if "API_KEY" in st.secrets else os.environ.get("API_KEY")
GOOGLE_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

st.set_page_config(page_title="Fake News Detector (Ensemble)", layout="wide")

# ==============================
# Trusted Sources (200+)
# ==============================
trusted_sources = [
    # Indian News
    "thehindu.com","timesofindia.com","hindustantimes.com","ndtv.com","indiatoday.in",
    "indianexpress.com","livemint.com","business-standard.com","deccanherald.com",
    "telegraphindia.com","mid-day.com","dnaindia.com","scroll.in","firstpost.com",
    "theprint.in","news18.com","oneindia.com","outlookindia.com","zeenews.india.com",
    "cnnnews18.com","economictimes.indiatimes.com","financialexpress.com","siasat.com",
    "newindianexpress.com","tribuneindia.com","asianage.com","bharattimes.com",
    "freepressjournal.in","morningindia.in","abplive.com","newsable.asianetnews.com",
    # International News
    "bbc.com","cnn.com","reuters.com","apnews.com","aljazeera.com","theguardian.com",
    "nytimes.com","washingtonpost.com","bloomberg.com","dw.com","foxnews.com","cbsnews.com",
    "nbcnews.com","abcnews.go.com","sky.com","france24.com","rt.com","sputniknews.com",
    "npr.org","telegraph.co.uk","thetimes.co.uk","independent.co.uk","globaltimes.cn",
    "china.org.cn","cbc.ca","abc.net.au","smh.com.au","japantimes.co.jp","lemonde.fr",
    "elpais.com","derstandard.at","spiegel.de","tagesschau.de","asiatimes.com",
    "straitstimes.com","thaiworldview.com","thejakartapost.com","thestandard.com.hk",
    "sbs.com.au","hawaiinewsnow.com","theglobeandmail.com","irishnews.com","latimes.com",
    "chicagotribune.com","startribune.com","nydailynews.com","financialtimes.com",
    "forbes.com","thehill.com","vox.com","buzzfeednews.com","huffpost.com","usatoday.com",
    "teleSURenglish.net","euronews.com","al-monitor.com","news.com.au","cnbc.com",
    "barrons.com","time.com","foreignpolicy.com","economist.com","foreignaffairs.com",
    "dailytelegraph.com.au","smh.com.au","thesun.co.uk","dailymail.co.uk",
    # Indian Government
    ".gov.in","pib.gov.in","isro.gov.in","pmindia.gov.in","mod.gov.in","mha.gov.in",
    "rbi.org.in","sebi.gov.in","nic.in","mohfw.gov.in","moef.gov.in","meity.gov.in",
    "railway.gov.in","dgca.gov.in","drdo.gov.in","indianrailways.gov.in","education.gov.in",
    "scienceandtech.gov.in","urbanindia.nic.in","financialservices.gov.in",
    "commerce.gov.in","sportsauthorityofindia.nic.in","agriculture.gov.in","power.gov.in",
    "parliamentofindia.nic.in","taxindia.gov.in","cbic.gov.in","epfindia.gov.in","defence.gov.in",
    # International Government & UN/NGO
    ".gov",".europa.eu","un.org","who.int","nasa.gov","esa.int","imf.org","worldbank.org",
    "fao.org","wto.org","unicef.org","unhcr.org","redcross.org","cdc.gov","nih.gov","usa.gov",
    "canada.ca","gov.uk","australia.gov.au","japan.go.jp","ec.europa.eu","consilium.europa.eu",
    "ecb.europa.eu","unep.org","ilo.org","ohchr.org","unodc.org","unwomen.org",
    "unfpa.org","unesco.org","wmo.int","ifrc.org","nato.int","oecd.org","europarl.europa.eu",
    "unido.org","wfp.org"
]

def is_trusted(url: str) -> bool:
    url = (url or "").lower()
    return any(src in url for src in trusted_sources)

# ==============================
# Text Cleaning
# ==============================
def clean_text(text: str) -> str:
    if not text:
        return ""
    # remove timestamps like "10 hours ago"
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|days|weeks|ago)\b", "", text, flags=re.I)
    # remove promotional / CTA
    text = re.sub(r"(share|save|click here|more details|read more|subscribe|follow us)", "", text, flags=re.I)
    # remove urls
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # remove emails
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # remove emojis / non text (keep common punctuation)
    text = re.sub(r"[^\w\s,.!?;:()'\"]+", "", text)
    # excessive punctuation
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    # remove long meaningless numbers
    text = re.sub(r"\b\d{5,}\b", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# ==============================
# Web Scraping
# ==============================
def scrape_url(url: str) -> str | None:
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
        combined = (title + "\n\n" + text)[:4000]
        return clean_text(combined)
    except Exception:
        return None

# ==============================
# Model loading 
# ==============================
LOCAL_MODELS = {"bert": None, "roberta": None, "flan": None}
MODEL_STATUS = {"bert": False, "roberta": False, "flan": False}

if TRANSFORMERS_AVAILABLE:
    @st.cache_resource
    def load_local_bert():
        try:
            model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
            tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
            pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
            MODEL_STATUS["bert"] = True
            return pipe
        except Exception as e:
            MODEL_STATUS["bert"] = False
            return None

    @st.cache_resource
    def load_local_roberta():
        try:
            pipe = pipeline("zero-shot-classification", model="roberta-large-mnli")
            MODEL_STATUS["roberta"] = True
            return pipe
        except Exception as e:
            MODEL_STATUS["roberta"] = False
            return None

    @st.cache_resource
    def load_local_flan():
        try:
            # Flan-T5 for explanation/generation (seq2seq)
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
            MODEL_STATUS["flan"] = True
            return gen
        except Exception as e:
            MODEL_STATUS["flan"] = False
            return None

    try:
        LOCAL_MODELS["bert"] = load_local_bert()
        LOCAL_MODELS["roberta"] = load_local_roberta()
        LOCAL_MODELS["flan"] = load_local_flan()
    except Exception:
        # if cache_resource fails, leave None and fall back later
        pass

# ==============================
# Remote API helpers (clear labeling)
# ==============================
def query_remote_classification(text: str) -> tuple[str, str]:
    """
    Calls remote generative API to classify and explain.
    Returns (classification, explanation)
    """
    if not API_KEY:
        return "UNSURE", "No API key configured for remote classification."
    url = f"{GOOGLE_GEMINI_ENDPOINT}?key={API_KEY}"
    payload = {
        "contents": [
            {"parts": [{"text": f"""Classify the following news as REAL or FAKE. 
Answer strictly with either 'REAL' or 'FAKE' on the first line. 
Then on the second line, give a short explanation (2–3 sentences) why you classified it that way.

Text:
{text}"""}]}
        ]
    }
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
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
    except Exception as e:
        return "ERROR", f"Remote API error: {e}"

def query_remote_correction(fake_text: str) -> str:
    if not API_KEY:
        return "No API key configured for remote correction."
    url = f"{GOOGLE_GEMINI_ENDPOINT}?key={API_KEY}"
    payload = {
        "contents": [
            {"parts": [{"text": f"""The following statement is FAKE. 
Please give the correct or factual version of it in one or two sentences.

Fake statement:
{fake_text}"""}]}
        ]
    }
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        correction = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        return correction
    except Exception as e:
        return f"No correction available (error: {e})"

# ==============================
# Ensemble Prediction
# ==============================
def predict_text_ensemble(text: str, url: str = "") -> tuple[str, dict]:
    """
    Returns final verdict and a dict with details:
    { 'method': 'local'|'remote'|'hybrid', 'bert':..., 'roberta':..., 'explanation':..., 'used_local': {...} }
    """
    text = clean_text(text)
    details = {"used_local": {}, "bert": None, "roberta": None, "flan_explanation": None, "method": None}

    # If the URL is trusted, short-circuit
    if url and is_trusted(url):
        details["method"] = "trusted_source"
        return "REAL", details

    # Try local models first (if present)
    bert_pipe = LOCAL_MODELS.get("bert")
    roberta_pipe = LOCAL_MODELS.get("roberta")
    flan_pipe = LOCAL_MODELS.get("flan")

    # If both local available, use local ensemble
    if bert_pipe and roberta_pipe:
        details["used_local"]["bert"] = True
        details["used_local"]["roberta"] = True
        try:
            bert_res = bert_pipe(text[:512])[0]['label']
            bert_pred = "REAL" if "REAL" in bert_res.upper() else "FAKE"
            details["bert"] = bert_res
        except Exception as e:
            bert_pred = None
            details["bert"] = f"error: {e}"

        try:
            roberta_res = roberta_pipe(text, candidate_labels=["REAL", "FAKE"])
            roberta_pred = roberta_res['labels'][0]
            details["roberta"] = roberta_res
        except Exception as e:
            roberta_pred = None
            details["roberta"] = f"error: {e}"

        # Weighted voting
        scores = {"REAL": 0.0, "FAKE": 0.0}
        for p, w in zip([bert_pred, roberta_pred], [0.5, 0.5]):
            if p == "REAL": scores["REAL"] += w
            elif p == "FAKE": scores["FAKE"] += w

        final = "REAL" if scores["REAL"] > scores["FAKE"] else ("FAKE" if scores["FAKE"] > scores["REAL"] else "UNSURE")
        details["method"] = "local"
        # Use flan for short explanation if available
        if flan_pipe:
            try:
                expl = flan_pipe(f"Explain in 1-2 sentences why this text is {final}: {text}", max_length=120)
                details["flan_explanation"] = expl[0].get("generated_text", "")
            except Exception:
                details["flan_explanation"] = None
        return final, details

    # If local unavailable, call remote API and be transparent
    details["method"] = "remote"
    classification, explanation = query_remote_classification(text)
    details["bert"] = None
    details["roberta"] = None
    details["flan_explanation"] = explanation
    return classification, details

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection App (DL Ensemble)")

st.markdown("""
**How this app runs (transparently):**
- It will *try* to load **local** models (BERT, RoBERTa, Flan-T5) and run the ensemble *locally* if possible.
- If local models are not available (e.g., you don't have weights / GPU), the app will *fall back* to a remote generative API and the UI will indicate that.
- This behavior is shown explicitly below so a reviewer knows which mode was used.
""")

input_type = st.radio("Input type", ["Text", "URL"])
user_input = ""
page_url = ""

if input_type == "Text":
    user_input = st.text_area("Enter news text here", height=200)
else:
    page_url = st.text_input("Enter news article URL")
    if page_url:
        scraped = scrape_url(page_url)
        if scraped:
            st.text_area("Extracted Article (from URL)", scraped, height=300)
            user_input = scraped
        else:
            st.warning("Could not scrape the URL or extracted text was empty.")

col1, col2 = st.columns([3, 1])
with col2:
    st.subheader("Model status")
    st.write("Local transformers available:", TRANSFORMERS_AVAILABLE)
    st.write("Local BERT loaded:", bool(LOCAL_MODELS.get("bert")))
    st.write("Local RoBERTa loaded:", bool(LOCAL_MODELS.get("roberta")))
    st.write("Local Flan-T5 loaded:", bool(LOCAL_MODELS.get("flan")))
    st.write("Remote API configured:", bool(API_KEY))

with col1:
    if st.button("Analyze"):
        if not (user_input and user_input.strip()):
            st.warning("Please enter text or a valid URL with content.")
        else:
            with st.spinner("Running analysis..."):
                verdict, details = predict_text_ensemble(user_input, page_url)
            st.subheader("Final Verdict")
            if verdict == "REAL":
                st.success("🟢 REAL NEWS")
            elif verdict == "FAKE":
                st.error("🔴 FAKE NEWS")
            elif verdict == "UNSURE":
                st.warning("⚠️ UNSURE")
            else:
                st.info(f"Result: {verdict}")

            st.markdown("**Analysis details:**")
            st.json(details)

            if details.get("flan_explanation"):
                st.markdown("**Explanation / Summary:**")
                st.write(details["flan_explanation"])

            with st.expander("🔎 Show Extracted / Cleaned Text"):
                st.write(user_input)

            # If remote and classification is FAKE, offer the correction helper
            if details.get("method") == "remote" and verdict == "FAKE":
                if st.button("Get short factual correction (remote)"):
                    correction = query_remote_correction(user_input)
                    st.markdown("**Suggested correction:**")
                    st.write(correction)

# Footer / notes
st.markdown("---")
st.markdown("**Notes:** Do not commit API keys to source control. Use `st.secrets` or environment variables. "
            "If you want the demo to *always* run locally for your final review, ensure you pre-download and place the model weights locally and run on a machine with sufficient memory/GPU.")

