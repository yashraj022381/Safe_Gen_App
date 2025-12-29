import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="India Helper AI Chatbot", page_icon="🇮🇳")
st.title("🇮🇳 भारत हेल्पर AI - आपकी समस्याओं का समाधान")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get Groq API key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ GROQ_API_KEY not found in Secrets. Add it in Settings → Secrets.")
    st.stop()

# Improved system prompt for natural Hindi + English
system_prompt = """
You are "Bharat Helper" - a friendly, caring AI assistant made for people in India.
- Always reply in the same language the user is using (Hindi, English, or Hinglish).
- If user writes in Hindi, reply in simple, natural Hindi (use Devanagari script properly).
- If user mixes Hindi-English (Hinglish), reply in easy Hinglish.
- Be empathetic, practical, and encouraging.
- Help with real Indian problems: jobs, education, farming, health, government schemes, money, family, etc.
- Keep answers short and clear unless user asks for details.
"""

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Welcome message on first load
if not st.session_state.messages:
    welcome = "नमस्ते! 👋 मैं भारत हेल्पर हूँ।\n\nआप किसी भी समस्या के बारे में हिंदी या अंग्रेजी में पूछ सकते हैं - नौकरी, पढ़ाई, खेती, सरकारी योजना, स्वास्थ्य, या कुछ भी।\n\nक्या मदद चाहिए आज?"
    st.session_state.messages.append(AIMessage(content=welcome))
    with st.chat_message("assistant"):
        st.markdown(welcome)

# User input
if prompt := st.chat_input("यहाँ अपनी समस्या लिखें... (हिंदी या अंग्रेजी में)"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("सोच रहा हूँ..."):
            llm = ChatGroq(
                model="llama-3.1-8b-instant",  # fast & good Hindi
                # model="llama-3.1-70b-versatile",  # even better Hindi if you want (slightly slower)
                api_key=groq_api_key,
                temperature=0.7
            )

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_input}"),
            ])

            chain = prompt_template | llm | StrOutputParser()

            chat_history_for_chain = st.session_state.messages[:-1]

            response = chain.invoke({
                "chat_history": chat_history_for_chain,
                "user_input": prompt
            })

            st.markdown(response)

    st.session_state.messages.append(AIMessage(content=response))
