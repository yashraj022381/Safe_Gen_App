import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set up page config
st.set_page_config(page_title="India Helper AI Chatbot", page_icon="🇮🇳")
st.title("🇮🇳 India Problem Solver AI Agent")

# === FIX STARTS HERE: Initialize session state FIRST ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get API key from secrets (for Streamlit Cloud)
api_key = st.secrets["GROQ_API_KEY"]

# System prompt
system_prompt = """
You are a helpful AI assistant focused on solving real-life problems for people in India.
Answer in simple English or Hindi if needed. Cover topics like jobs, education, farming, health, government schemes, etc.
Be empathetic and practical.
"""

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
if prompt := st.chat_input("Ask me anything about problems in India..."):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            chain = (
                {"input": RunnablePassthrough(), 
                 "chat_history": lambda x: st.session_state.messages[:-1]}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            response = chain.invoke(prompt)
            st.write(response)

    # Add AI response to history
    st.session_state.messages.append(AIMessage(content=response))
