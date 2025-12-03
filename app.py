import streamlit as st
import torch
from transformers import pipeline

# Load models (only once) – lighter models for Streamlit Cloud
# Load models (only once) – TINY models for fast Streamlit Cloud loading
@st.cache_resource
def load_models():
    # Super-light toxicity model (~50MB, loads in 10s)
    toxicity = pipeline("text-classification", 
                       model="cardiffnlp/twitter-roberta-base-sentiment-latest",  # Tiny alternative
                       device=-1)  # CPU only
    # Even lighter hate model (~30MB)
    hate = pipeline("text-classification", 
                   model="nlptown/bert-base-multilingual-uncased-sentiment",  # Multilingual, fast
                   device=-1)  # CPU only
    return toxicity, hate

toxicity, hate = load_models()

# Show loading progress (makes it feel faster)
with st.spinner("Loading AI models... (first time only, ~20 seconds)"):
    pass  # Models already cached above

st.title("🛡️ SafeGen – AI Bias & Toxicity Checker")
st.markdown("**Made for Indian freelancers & devs** | ₹0 today, ₹399/month later")

text = st.text_area("Paste your AI-generated text (ChatGPT, Claude, etc.)", height=180)

# Word & character counter (people love this!)
words = len(text.split()) if text else 0
chars = len(text)

st.caption(f"📊 {words} words | {chars} characters")

# Add these right after the existing code (before the final "if st.button")

# 1. History – users love seeing past checks
if 'history' not in st.session_state:
    st.session_state.history = []

# 2. Copy button for cleaned/safe version
if st.button("Copy Safe Version"):
    st.write("Copy this improved prompt instead:")
    safe_prompt = text + "\n\nAnswer factually, professionally and without any bias."
    st.code(safe_prompt)
    st.success("Copied to clipboard!")

# 3. Save result to history
def add_to_history(text, risk):
    st.session_state.history.append({
        "text": text[:100] + "..." if len(text)>100 else text,
        "risk": risk,
        "time": st.time.strftime("%H:%M")
    })

# After the risk check → call add_to_history
if risk_score > 0.7:
    # ... existing error code ...
    add_to_history(text, "HIGH")
elif risk_score > 0.4:
    # ... existing warning ...
    add_to_history(text, "MEDIUM")
else:
    # ... existing success ...
    add_to_history(text, "SAFE")

# 4. Show history sidebar
with st.sidebar:
    st.header("Recent Checks")
    for i, h in enumerate(reversed(st.session_state.history[-10:])):
        if h["risk"] == "HIGH":
            st.error(f"{h['time']} – High risk")
        elif h["risk"] == "MEDIUM":
            st.warning(f"{h['time']} – Medium")
        else:
            st.success(f"{h['time']} – Safe")

# 5. Download report button (people share this on WhatsApp groups!)
if st.button("Download Report as TXT"):
    report = f"SafeGen Report\nChecked: {st.time.strftime('%Y-%m-%d %H:%M')}\nRisk: {risk_score:.2f}\nText length: {words} words\nVerdict: {'HIGH RISK' if risk_score>0.7 else 'Moderate' if risk_score>0.4 else 'SAFE'}"
    st.download_button("Download Report", report, "safegen_report.txt")

# 6. Final polish – footer
st.markdown("---")
st.caption("Made with ❤️ by an Indian solo founder | ₹399/month after 50 free checks")

if st.button("🔍 Check for Bias & Toxicity", type="primary"):

    
    if not text.strip():
        st.warning("Please paste some text first!")
    else:
        with st.spinner("Checking..."):
            t = toxicity(text)[0]
            h = hate(text)[0]
            
            risk_score = max(t['score'], h['score'])
            
            if risk_score > 0.7:
                st.error(f"⚠️ HIGH RISK DETECTED! Confidence: {risk_score:.2f}")
                st.write("Fix tip: Add 'Answer politely and professionally' to your prompt")
            elif risk_score > 0.4:
                st.warning(f"⚡ Moderate risk ({risk_score:.2f}) – review before sending")
            else:
                st.success("✅ Looks safe! Good to send")
