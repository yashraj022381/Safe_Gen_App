
import streamlit as st
from transformers import pipeline

# Tiny & fast models that NEVER fail on Streamlit free
toxicity = pipeline("text-classification", model="martin-ha/toxic-comment-model")   # tiny & fast
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

st.set_page_config(page_title="SafeGen", page_icon="Shield")

st.title("SafeGen – Hallucination & Bias Checker")
st.caption("Made for Indian freelancers & small teams | Free to try | ₹399/month later")

text = st.text_area("Paste any AI-generated text (ChatGPT, Gemini, etc.)", height=150)

if st.button("Check Safety", type="primary"):
    with st.spinner("Checking..."):
        tox = toxicity(text)[0]
        sent = sentiment(text)[0]

        risk_score = 0
        if tox['label'] == 'toxic' and tox['score'] > 0.6:
            risk_score += 1
        if sent['label'] in ['NEGATIVE', 'neg']:
            risk_score += 0.5

        if risk_score >= 1:
            st.error(f"High Risk Detected! (Score: {risk_score:.1f}/1.5)")
            st.write("Fix: Add 'Answer politely and professionally' to your prompt")
        elif risk_score >= 0.5:
            st.warning(f"Medium Risk (Score: {risk_score:.1f}/1.5)")
            st.write("Consider rephrasing for safer tone")
        else:
            st.success("Safe & Professional!")
            st.balloons()
st.markdown("---")
st.markdown("Built by an Indian solo founder • [Follow on X](https://x.com/yourusername) • ₹399/month after 50 free checks")
