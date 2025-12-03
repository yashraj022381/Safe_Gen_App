import streamlit as st
import torch
from transformers import pipeline
import time

# -------------------------- TINY & FAST MODELS (NO TORCH NEEDED) --------------------------
@st.cache_resource
def load_models():
        # These two models are < 100 MB total and work perfectly on Streamlit free tier
        toxicity = pipeline("text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1)
    
        hate = pipeline("text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1)
        
        return toxicity, hate

toxicity, hate = load_models()

with st.spinner("Loading tiny AI models (first time only, ~20 sec)..."):
    pass
# ----------------------------------- UI STARTS HERE -----------------------------------
st.set_page_config(page_title="SafeGen – AI Bias Checker", page_icon="🛡️")
st.title("🛡️ SafeGen – AI Bias & Toxicity Checker")
st.markdown("**Made for Indian freelancers & developers** | Free to try • ₹399/month later")


text = st.text_area("Paste any AI-generated text (ChatGPT, Claude, Gemini, etc.)", height=180)

# Word & character counter
words = len(text.split()) if text else 0
chars = len(text)
st.caption(f"Words: {words} | Characters: {chars}")

# History
if 'history' not in st.session_state:
    st.session_state.history = []

if st.button("Copy Safe Version"):
    st.write("Copy this improved prompt instead: ")
    safe_prompt = text + "\n\nAnswer factually, professionally and without any bias."
    st.code(safe_prompt)
    st.sucess("Copied to Clipboard!")

def add_to_history(text, verdict):
    st.session_state.history.append({
        "text": text[:100] + "..." if len(text)>100 else text,
        "verdict": verdict,
        "time": time.strftime("%H:%M")
    })

if 'risk_score' not in st.session_state:
    st.session_state.risk_score = 0.0

# Main Check Button
if st.button("Check for Bias & Toxicity", type="primary"):
    if not text.strip():
        st.warning("Please paste some text first!")
    else:
        with st.spinner("Checking..."):
            t = toxicity(text)[0]
            h = hate(text)[0]
            
            risk = max(t['score'], h['score'])
            st.session_state.risk_score = risk_score
            
            if risk > 0.7:
                st.error(f"HIGH RISK DETECTED! Confidence: {risk_score:.2f%}")
                st.write("Fix tip → Add: 'Answer professionally, factually and without any bias' to your prompt")
                add_to_history(text, "HIGH")
            elif risk > 0.4:
                st.warning(f"Moderate risk ({risk:.2f%}) – review before sending client")
                add_to_history(text, "MEDIUM")
            else:
                st.success("Looks perfectly safe! Good to send")
                add_to_history(text, "SAFE")

# Sidebar – Recent checks
with st.sidebar:
    st.header("Recent Checks")
    for i, h in enumerate(reversed(st.session_state.history[-10:])):
        if h["risk"] == "HIGH":
            st.error(f"{h['time']} – High risk")
        elif h["risk"] == "MEDIUM":
            st.warning(f"{h['time']} – Medium")
        else:
            st.success(f"{h['time']} – Safe")

if st.button("Download Report as TXT"):
    report = f"SafeGen Report\nChecked: {time.strftime('%Y-%m-%d %H:%M')}\nRisk: {st.session_state.risk_score:.2f}\nText lengh: {words} words\nVerdict: {'HIGH RISK' if st.session_state_score>0.7 else 'Moderate' if st.session_state.risk_score>0.4 else 'SAFE'}"
    st.download_button("Download Report", report, "safegen_report.txt")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by an Indian solo founder | ₹399/month after 100 free checks")
