import streamlit as st
from transformers import pipeline
import time
import torch

# ───── TINY & FAST MODELS (load in 15–25 seconds on free tier) ─────
@st.cache_resource
def load_models():
    with st.spinner("First time loading tiny AI models… (15–25 sec only once)"):
        # Super lightweight toxicity + bias detector
        toxicity = pipeline(
            "text-classification",
            model="lxyuan/distilbert-base-multilingual-cased-toxicity",
            device=-1  # CPU only
        )
        # Extra sentiment/hate check (tiny model)
        hate = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1
        )
    return toxicity, hate

toxicity, hate = load_models()

# ───── HISTORY & SESSION STATE ─────
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(text, verdict):
    st.session_state.history.append({
        "text": text[:80] + "..." if len(text)>80 else text,
        "verdict": verdict,
        "time": time.strftime("%H:%M")
    })

# ───── MAIN UI ─────
st.title("SafeGen – AI Bias & Toxicity Checker")
st.markdown("**Made for Indian freelancers & devs** | Free trial → ₹399/month later")

text = st.text_area("Paste any AI output (ChatGPT, Claude, Gemini…)", height=180)

words = len(text.split()) if text else 0
chars = len(text)
st.caption(f"Words: {words} | Characters: {chars}")

col1, col2 = st.columns(2)
with col1:
    check = st.button("Check for Bias & Toxicity", type="primary")
with col2:
    copy_safe = st.button("Copy Safe Version")

if check and text:
    with st.spinner("Checking…"):
        t = toxicity(text)[0]
        h = hate(text)[0]
        score = max(t['score'], h['score'] if 'score' in h else 0)

        if score > 0.75:
            st.error(f"HIGH RISK! Confidence: {score:.2f}")
            verdict = "HIGH RISK"
        elif score > 0.45:
            st.warning(f"Moderate risk – review before sending ({score:.2f})")
            verdict = "MODERATE"
        else:
            st.success("SAFE – Good to send to client!")
            verdict = "SAFE"
        
        add_to_history(text, verdict)

if copy_safe and text:
    safe_text = text + "\n\n[Add this line → Answer factually, professionally and without any bias or toxic language.]"
    st.code(safe_text)
    st.success("Copied! Paste this improved version")

# ───── SIDEBAR HISTORY ─────
with st.sidebar:
    st.header("Recent Checks")
    for h in reversed(st.session_state.history[-10:]):
        if h["verdict"] == "HIGH RISK":
            st.error(f"{h['time']} – High risk")
        elif h["verdict"] == "MODERATE":
            st.warning(f"{h['time']} – Moderate")
        else:
            st.success(f"{h['time']} – Safe")

# ───── FOOTER ─────
st.markdown("---")
st.caption("Built in 2 days by an Indian solo founder ❤️ | ₹399/month after 100 free checks")
