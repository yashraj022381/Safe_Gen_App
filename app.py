import streamlit as st
from transformers import pipeline
import time

# -------------------------- TINY & FAST MODELS (NO TORCH NEEDED) --------------------------
@st.cache_resource
def load_models():
    with st.spinner("Loading tiny AI models (first time only, ~20 sec)..."):
        # These two models are < 100 MB total and work perfectly on Streamlit free tier
        toxicity = pipeline(
            "text-classification",
            model="lxyuan/distilbert-base-multilingual-cased-toxicity",
            device=-1  # CPU only
        )
        hate = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device=-1
        )
        toxicity, hate = load_models()
    return toxicity, hate

#toxicity, hate = load_models()

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

def add_to_history(text, verdict):
    st.session_state.history.append({
        "text": text[:80] + "..." if len(text)>80 else text,
        "verdict": verdict,
        "time": time.strftime("%H:%M")
    })

# Main Check Button
if st.button("Check for Bias & Toxicity", type="primary"):
    if not text.strip():
        st.warning("Please paste some text first!")
    else:
        with st.spinner("Analysing..."):
            t = toxicity(text)[0]
            h = hate(text)[0]
            risk = max(t['score'], h['score'])

            if risk > 0.85:
                st.error(f"HIGH RISK DETECTED! Confidence: {risk:.2%}")
                st.write("Fix tip → Add: 'Answer professionally, factually and without any bias' to your prompt")
                add_to_history(text, "HIGH")
            elif risk > 0.5:
                st.warning(f"Moderate risk ({risk:.2%}) – review before sending client")
                add_to_history(text, "MEDIUM")
            else:
                st.success("Looks perfectly safe! Good to send")
                add_to_history(text, "SAFE")

        # Download report
        report = f"SafeGen Report\nDate: {time.strftime('%Y-%m-%d %H:%M')}\nRisk: {risk:.2%}\nVerdict: {'HIGH' if risk>0.85 else 'MEDIUM' if risk>0.5 else 'SAFE'}\n\nText:\n{text}"
        st.download_button("Download Report (.txt)", report, "safegen_report.txt")

# Sidebar – Recent checks
with st.sidebar:
    st.header("Recent Checks")
    for item in reversed(st.session_state.history[-8:]):
        if item["verdict"] == "HIGH":
            st.error(f"{item['time']} – High risk")
        elif item["verdict"] == "MEDIUM":
            st.warning(f"{item['time']} – Medium")
        else:
            st.success(f"{item['time']} – Safe")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by an Indian solo founder | ₹399/month after 100 free checks")
