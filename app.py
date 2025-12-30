import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from streamlit_mic_recorder import mic_recorder



st.set_page_config(page_title="Bharat Helper AI Chatbot", page_icon="🇮🇳")
st.title("🇮🇳 भारत हेल्पर\Bharat Helper AI - आपकी समस्याओं का समाधान")
# Sidebar info
st.sidebar.markdown(r"## 🇮🇳 भारत हेल्पर\Bharat Helper AI")
st.sidebar.markdown("It was created for Indain peoples to solve their daily life problems in easier way.\n\nयह AI भारत के लोगों की रोज़मर्रा की समस्याओं में मदद करने के लिए बनाया गया है।")
st.sidebar.markdown(r"**बनाया गया\Created by:** Yashraj")
st.sidebar.markdown(r"**सपोर्ट\Support:** your.email@gmail.com")
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Groq + Llama 3.1 ⚡")
st.sidebar.markdown(r"**🌍 भाषाएँ:** हिंदी, मराठी, বাংলা, ਪੰਜਾਬੀ, தமிழ், తెలుగు और अधिक")

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
system_prompt = """You are "Bharat Helper" - a friendly and reliable AI for people from every corner of India.
- Respond in the same language the user uses (Hindi, Marathi, Bengali, Punjabi, Tamil, Telugu, Gujarati, Kannada, Malayalam, Bhojpuri, Haryanvi, etc.).
- If the language is mixed, respond in the same mixed language.
- The response should be simple, concise, and heartfelt.
- Topics: Jobs, education, farming, government schemes, health, family, money, daily life, etc.
- Always try to help and offer encouragement."""

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# ... (after display history)

# Mic input button
audio = mic_recorder(start_prompt="🎤 Start recording", stop_prompt="🛑 Stop", key='recorder')

if audio:
    # Save audio to temp file
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio['bytes'])

    # Transcribe with Groq Whisper (add your Groq key if not already)
    from groq import Groq
    client = Groq(api_key=groq_api_key)
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            response_format="text",
            language="hi" if "hindi" in prompt.lower() else "en"  # Auto-detect or set
        )
    prompt = transcription  # Use transcribed text as input

    # Then proceed with adding to messages and generating response as before
            
 # Add a "Clear Chat" button in sidebar
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()  # Refresh the page           

# Welcome message on first load
if not st.session_state.messages:
    welcome = """नमस्ते! 🙏  
নমস্কার! | नमस्कार! | ਸਤ ਸ੍ਰੀ ਅਕਾਲ! | નમસ્તે!  
வணக்கம்! | నమస్కారం! | नमस्कार!

मैं भारत हेल्पर हूँ।  
आप अपनी मातृभाषा में कोई भी समस्या पूछ सकते हैं।  
नौकरी, पढ़ाई, खेती, सरकारी योजना, स्वास्थ्य - सबके लिए मदद करता हूँ।

आज आपकी क्या मदद करूँ? 😊 \
    \n\t Hello! 👋 I am Bharat Helper.\n\nI can help you in any problem you tell me that on in Hindi, Marathi or English or in any language you like you can ask - Jobs, Study, Farming, Government Schemes, Health, all most anything. \n \n So what help do you want to today?"""
    st.session_state.messages.append(AIMessage(content=welcome))
    with st.chat_message("assistant"):
        st.markdown(welcome)

# User input
if prompt := st.chat_input("Write down your problem... (in any language)\n\nयहाँ अपनी समस्या लिखें... (हिंदी, मराठी, বাংলা, ਪੰਜਾਬੀ, தமிழ் आदि)"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("I am thinking...\nसोच रहा हूँ..."):
            llm = ChatGroq(
                model="llama-3.1-8b-instant",  # fast & good Hindi
                # model="llama-3.1-70b-versatile",  # even better Hindi if you want (slightly slower)
                api_key=groq_api_key,
                temperature=0.7
            )


           # Agent prompt for reasoning + tools
            #agent_prompt = PromptTemplate.from_template("""
             #{system_prompt}
    
              #You have access to tools. Use them only if needed for the query.
    
              #Chat history: {chat_history}
              #User input: {user_input}
            #""")

           # Create agent
           # agent = create_react_agent(llm, tools, agent_prompt)
            #agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_input}")
            ])

            chain = prompt_template | llm | StrOutputParser()

            chat_history_for_chain = st.session_state.messages[:-1]

            final_response = chain.invoke({
                "chat_history": chat_history_for_chain,
                "user_input": prompt 
            })
            st.markdown(final_response)
    st.session_state.messages.append(AIMessage(content=final_response))
    #st.session_state.messages.append(AIMessage(content=response))
            # Invoke agent with history
           # input_data = {
            #    "system_prompt": system_prompt,
             #   "chat_history": "\n".join([msg.content for msg in st.session_state.messages[:-1]]),
              #  "user_input": prompt
            #}
            #response = agent_executor.invoke(input_data)["output"]

            
