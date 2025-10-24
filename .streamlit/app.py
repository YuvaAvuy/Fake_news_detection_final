import os
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

API_KEY = st.secrets.get("API_KEY") if "API_KEY" in st.secrets else os.environ.get("API_KEY")
GOOGLE_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

st.set_page_config(page_title="Fake News Detector", layout="wide")

# Hide Streamlit default UI (hamburger, footer, menu)
hide_streamlit_style = """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton, div[data-testid="stToolbar"], div[data-testid="stStatusWidget"] {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# =========================================
# Trusted sources list
# =========================================
trusted_sources = [
    # Indian News
    "thehindu.com","timesofindia.com","hindustantimes.com","ndtv.com","indiatoday.in", "indianexpress.com","livemint.com","business-standard.com","deccanherald.com", "telegraphindia.com","mid-day.com","dnaindia.com","scroll.in","firstpost.com", "theprint.in","news18.com","oneindia.com","outlookindia.com","zeenews.india.com", "cnnnews18.com","economictimes.indiatimes.com","financialexpress.com","siasat.com", "newindianexpress.com","tribuneindia.com","asianage.com","bharattimes.com", "freepressjournal.in","morningindia.in","abplive.com","newsable.asianetnews.com", # International News 
    "bbc.com","cnn.com","reuters.com","apnews.com","aljazeera.com","theguardian.com", "nytimes.com","washingtonpost.com","bloomberg.com","dw.com","foxnews.com","cbsnews.com", "nbcnews.com","abcnews.go.com","sky.com","france24.com","rt.com","sputniknews.com", "npr.org","telegraph.co.uk","thetimes.co.uk","independent.co.uk","globaltimes.cn", "china.org.cn","cbc.ca","abc.net.au","smh.com.au","japantimes.co.jp","lemonde.fr", "elpais.com","derstandard.at","spiegel.de","tagesschau.de","asiatimes.com", "straitstimes.com","thaiworldview.com","thejakartapost.com","thestandard.com.hk", "sbs.com.au","hawaiinewsnow.com","theglobeandmail.com","irishnews.com","latimes.com", "chicagotribune.com","startribune.com","nydailynews.com","financialtimes.com", "forbes.com","thehill.com","vox.com","buzzfeednews.com","huffpost.com","usatoday.com", "teleSURenglish.net","euronews.com","al-monitor.com","news.com.au","cnbc.com", "barrons.com","time.com","foreignpolicy.com","economist.com","foreignaffairs.com", "dailytelegraph.com.au","smh.com.au","thesun.co.uk","dailymail.co.uk", # Indian Government
    ".gov.in","pib.gov.in","isro.gov.in","pmindia.gov.in","mod.gov.in","mha.gov.in", "rbi.org.in","sebi.gov.in","nic.in","mohfw.gov.in","moef.gov.in","meity.gov.in", "railway.gov.in","dgca.gov.in","drdo.gov.in","indianrailways.gov.in","education.gov.in", "scienceandtech.gov.in","urbanindia.nic.in","financialservices.gov.in", "commerce.gov.in","sportsauthorityofindia.nic.in","agriculture.gov.in","power.gov.in", "parliamentofindia.nic.in","taxindia.gov.in","cbic.gov.in","epfindia.gov.in","defence.gov.in", # International Government & UN/NGO 
    ".gov",".europa.eu","un.org","who.int","nasa.gov","esa.int","imf.org","worldbank.org", "fao.org","wto.org","unicef.org","unhcr.org","redcross.org","cdc.gov","nih.gov","usa.gov", "canada.ca","gov.uk","australia.gov.au","japan.go.jp","ec.europa.eu","consilium.europa.eu", "ecb.europa.eu","unep.org","ilo.org","ohchr.org","unodc.org","unwomen.org", "unfpa.org","unesco.org","wmo.int","ifrc.org","nato.int","oecd.org","europarl.europa.eu", "unido.org","wfp.org"
]

def is_trusted(url: str) -> bool:
    url = (url or "").lower()
    return any(src in url for src in trusted_sources)

# =========================================
# Clean text
# =========================================
def clean_text(text: str) -> str:
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|days|weeks|ago)\b", "", text, flags=re.I)
    text = re.sub(r"(share|read more|subscribe|follow us)", "", text, flags=re.I)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s,.!?;:()'\"]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# =========================================
# Web scraping
# =========================================
def scrape_url(url: str) -> str | None:
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        article = soup.find("article") or soup.find("div", {"class": "articlebodycontent"})
        text_parts = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split()) > 5]
        text = " ".join(text_parts)
        return clean_text((title + "\n\n" + text)[:4000])
    except Exception:
        return None

# =========================================
# Local / remote hybrid
# =========================================
LOCAL_MODELS = {"bert": None, "flan": None}

if TRANSFORMERS_AVAILABLE:
    @st.cache_resource
    def load_local_bert():
        try:
            model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
            tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except:
            return None

    @st.cache_resource
    def load_local_flan():
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            return pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        except:
            return None

    LOCAL_MODELS["bert"] = load_local_bert()
    LOCAL_MODELS["flan"] = load_local_flan()

def query_remote_classification(text: str) -> tuple[str, str]:
    if not API_KEY:
        return "UNSURE", "No API key configured."
    url = f"{GOOGLE_GEMINI_ENDPOINT}?key={API_KEY}"
    body = {
        "contents": [{
            "parts": [{"text": f"Classify the following news as REAL or FAKE. "
                               f"Respond with 'REAL' or 'FAKE' and explain briefly:\n\n{text}"}]
        }]
    }
    try:
        resp = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=30)
        data = resp.json()
        raw = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        lines = raw.split("\n", 1)
        verdict = lines[0].strip().upper() if lines else "UNSURE"
        expl = lines[1].strip() if len(lines) > 1 else ""
        if "FAKE" in verdict: return "FAKE", expl
        if "REAL" in verdict: return "REAL", expl
        return "UNSURE", expl
    except Exception as e:
        return "UNSURE", f"Error: {e}"

# =========================================
# Ensemble classification
# =========================================
def predict_text_ensemble(text: str, url: str = "") -> tuple[str, str]:
    text = clean_text(text)
    if url and is_trusted(url):
        return "REAL", "Published by a verified source."
    bert = LOCAL_MODELS.get("bert")
    flan = LOCAL_MODELS.get("flan")
    if bert:
        try:
            label = bert(text[:512])[0]['label']
            verdict = "REAL" if "REAL" in label.upper() else "FAKE"
            explanation = ""
            if flan:
                expl = flan(f"Explain why this text is {verdict}: {text}", max_length=120)
                explanation = expl[0].get("generated_text", "")
            return verdict, explanation
        except:
            pass
    return query_remote_classification(text)

# =========================================
# Streamlit app UI (cleaned)
# =========================================
st.title("📰 Fake News Detector")

choice = st.radio("Choose Input Type", ["Text", "URL"])
user_input, url_input = "", ""

if choice == "Text":
    user_input = st.text_area("Enter news text", height=250)
else:
    url_input = st.text_input("Enter article URL")
    if url_input:
        scraped_text = scrape_url(url_input)
        if scraped_text:
            user_input = scraped_text
        else:
            st.warning("Could not retrieve article text.")

if st.button("Check News"):
    if not user_input.strip():
        st.warning("Please provide text or URL content.")
    else:
        verdict, explanation = predict_text_ensemble(user_input, url_input)
        if verdict == "REAL":
            st.success("🟢 This news appears REAL.")
        elif verdict == "FAKE":
            st.error("🔴 This news appears FAKE.")
        else:
            st.warning("⚠️ Unable to classify.")
        if explanation:
            st.write(explanation)
