import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
import base64
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice

# Page config
st.set_page_config(page_title="भारत हेल्पर AI", page_icon="🇮🇳")
st.title("🇮🇳 Bharat Helper AI\भारत हेल्पर AI - बोलकर पूछो, सुनकर जवाब पाओ 🔊")

# Sidebar
st.sidebar.markdown(r"**# 🇮🇳 भारत हेल्पर AI\Bharat Helper AI**")
st.sidebar.markdown(r"**🌟 बनाया\Created:** Yashraj")
st.sidebar.markdown(r"**📧 सपोर्ट\Support:** your.email@gmail.com")
st.sidebar.markdown(r"**🔊 Voice In & Out:** Groq Whisper + ElevenLabs Female")

# Clear chat
if st.sidebar.button("🗑️ चैट हिस्ट्री साफ़ करें\Clear chat history"):
    st.session_state.messages = []
    st.rerun()

# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

# API Keys
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=groq_api_key)
except:
    st.error("⚠️ GROQ_API_KEY नहीं मिला।")
    st.stop()

try:
    elevenlabs_api_key = st.secrets["ELEVENLABS_API_KEY"]
    eleven_client = ElevenLabs(api_key=elevenlabs_api_key)
except:
    st.warning("🔇 Voice output off - Add ELEVENLABS_API_KEY for female voice")
    eleven_client = None

# System prompt
system_prompt = """आप "भारत हेल्पर" हैं - भारत के हर कोने के लोगों के लिए एक दोस्ताना और भरोसेमंद AI।
- यूजर जो भी भाषा बोले या लिखे, उसी में जवाब दें (हिंदी, मराठी, बंगाली, पंजाबी, तमिल आदि)।
- जवाब छोटा, स्पष्ट और हौसला देने वाला हो|\n\nYou are "Bharat Helper" - a friendly and reliable AI for people from every corner of India.
- Respond in the same language the user speaks or writes (Hindi, Marathi, Bengali, Punjabi, Tamil, etc.).
- The response should be short, clear, and encouraging."""

# Welcome
if not st.session_state.messages:
    welcome = """नमस्ते! 🙏  
अब आप बोलकर भी पूछ सकते हैं! 🎤  
माइक बटन दबाएं → अपनी भाषा में बोलें → मैं सुनकर जवाब दूँगी 🔊

कोई भी समस्या पूछिए - नौकरी, पढ़ाई, खेती, स्वास्थ्य, सरकारी योजना...\n\nHello! 🙏
Now you can ask questions by speaking! 🎤
Press the microphone button → Speak in your language → I will listen and reply 🔊

Ask about any problem – jobs, studies, farming, health, government schemes..."""
    st.session_state.messages.append(AIMessage(content=welcome))
    with st.chat_message("assistant"):
        st.markdown(welcome)

# Show history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# === MIC VOICE INPUT ===
audio_bytes = st.experimental_audio_input("🎤 अपनी भाषा में बोलें\nSpeak in your own language")

prompt = None
if audio_bytes:
    with st.spinner("आपकी बात सुन रही हूँ\nI'm listening to you..."):
        # Save temp file
        with open("temp_voice.wav", "wb") as f:
            f.write(audio_bytes.getvalue())

        # Transcribe with Groq Whisper (excellent for Indian languages)
        with open("temp_voice.wav", "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=( "temp_voice.wav", file.read()),
                model="whisper-large-v3",
                response_format="text",
                language=None  # Auto-detect
            )
        prompt = transcription.text
        st.info(f"आपने कहा: **{prompt}**")

# === TEXT INPUT FALLBACK ===
if not prompt:
    prompt = st.chat_input("या यहाँ लिखें... (हिंदी, मराठी, বাংলা, ਪੰਜਾਬੀ आदि)\nOr write here... (Hindi, Marathi, Bengali, Punjabi etc.)")

# Process if there's input
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("जवाब दे रही हूँ...\nI'm answering..."):
            llm = ChatGroq(
                model="llama-3.1-70b-versatile",
                api_key=groq_api_key,
                temperature=0.6
            )

            template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            chain = template | llm | StrOutputParser()
            history = st.session_state.messages[:-1]

            response = chain.invoke({
                "chat_history": history,
                "input": prompt
            })

            st.markdown(response)

            # === FEMALE VOICE OUTPUT ===
            if eleven_client:
                try:
                    voice = Voice(voice_id="21m00Tcm4TlvDq8ikWAM")  # Rachel - natural female
                    audio_stream = eleven_client.generate(
                        text=response,
                        voice=voice,
                        model="eleven_multilingual_v2"
                    )
                    audio_bytes = b"".join(list(audio_stream))
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f"""
                    <audio controls autoplay>
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                    st.caption("🔊 मैं बोल रही हूँ!")
                except Exception as e:
                    st.caption("Voice error")

    st.session_state.messages.append(AIMessage(content=response))
