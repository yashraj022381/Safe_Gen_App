import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Page setup – must be first
st.set_page_config(page_title="India Helper AI Chatbot", page_icon="🇮🇳")
st.title("🇮🇳 India Problem Solver AI Agent")

# Initialize chat history early and safely
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get Groq API key from secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ Please add your GROQ_API_KEY in Secrets (Settings → Secrets) on Streamlit Cloud.")
    st.stop()

# System prompt – customize for India
system_prompt = """
You are a friendly and helpful AI assistant for people in India.
Answer in simple English or Hindi (if the user asks in Hindi).
Help with real problems: jobs, education, farming, health, government schemes, daily life, etc.
Be kind, practical, and positive.
"""

# Display previous messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
if prompt := st.chat_input("Ask me anything about problems in India..."):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking fast with Groq..."):
            # Use Groq model (llama3-8b is fast & good, or use llama3-70b for better quality)
            llm = ChatGroq(
                model="llama-3.1-8b-instant",  # or "llama-3.1-70b-versatile"
                api_key=groq_api_key,
                temperature=0.7
            )

            # Prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_input}"),
            ])

            # Chain – built fresh inside the block to avoid session state errors
            chain = prompt_template | llm | StrOutputParser()

            # Prepare history (all messages except the last user one)
            chat_history_for_chain = st.session_state.messages[:-1]

            # Invoke
            response = chain.invoke({
                "chat_history": chat_history_for_chain,
                "user_input": prompt
            })

            # Show response
            st.markdown(response)

    # Save AI response
    st.session_state.messages.append(AIMessage(content=response))
