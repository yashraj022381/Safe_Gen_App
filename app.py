import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# === MUST BE AT THE VERY TOP ===
st.set_page_config(page_title="India Helper AI Chatbot", page_icon="🇮🇳")
st.title("🇮🇳 India Problem Solver AI Agent")

# Initialize session state SAFELY and EARLY
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get API key securely (for Streamlit Cloud)
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("GROQ API key not found. Please add it in Secrets on Streamlit Cloud.")
    st.stop()

# System prompt - customize as needed
system_prompt = """
You are a helpful AI assistant focused on solving real-life problems for people in India.
Answer in simple English or Hindi if the user asks in Hindi. 
Cover topics like jobs, education, farming, health, government schemes, daily life, etc.
Be empathetic, practical, and positive.
"""

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Ask me anything about problems in India..."):
    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show thinking spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create LLM
            llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=api_key)

            # Define prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            # Create chain - IMPORTANT: pass history correctly without lambda bug
            chain = (
                {
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: st.session_state.messages[:-1]  # all except latest (user message)
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )

            # Invoke chain with just the user input
            response = chain.invoke({"input": prompt})

            # Display response
            st.markdown(response)

    # Add AI response to history
    st.session_state.messages.append(AIMessage(content=response))
