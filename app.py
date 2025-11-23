
import streamlit as st
import re
from textblob import TextBlob  # Simple sentiment (built-in, no downloads)

st.set_page_config(page_title="SafeGen", page_icon="🛡️")

st.title("🛡️ SafeGen – AI Output Safety Checker")
st.caption("Built for Indian freelancers & small teams | Free to try | ₹399/month later")

text = st.text_area("Paste any AI-generated text (ChatGPT, Gemini, etc.)", height=150)
if st.button("🔍 Check Safety", type="primary"):
    with st.spinner("Analyzing..."):
        # Simple toxicity rules (no model downloads – always works!)
        toxic_words = re.findall(r'\b(idiot|garbage|sucks|stupid|hate|trash|worst|loser|dumb)\b', text.lower())
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity  # -1 (negative) to +1 (positive)
        
        risk_score = len(toxic_words) + (1 if sentiment_score < -0.2 else 0)  # Simple scoring
        
        if risk_score >= 2:
            st.error(f"🚨 High Risk – Toxic/Biased! (Score: {risk_score}/3)")
            st.write("⚠️ Don't send to clients – risk of backlash!")
            st.write("💡 Fix: Prompt with 'Be polite, inclusive, and factual'")
        elif risk_score >= 1:
            st.warning(f"⚠️ Medium Risk (Score: {risk_score}/3) – Rephrase for safety")
        else:
            st.success("✅ Safe & Professional!")
            st.balloons()

st.markdown("---")
st.markdown("Solo Indian founder • Free for first 100 checks • [Follow on X](https://x.com/yourhandle)")
