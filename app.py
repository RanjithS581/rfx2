import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from textblob import TextBlob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd

os.environ["GOOGLE_API_KEY"] = "AIzaSyDeCfka3goQtaqdQFQ6IDW2qkdneps-Otw"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "preferences" not in st.session_state:
    st.session_state.preferences = {}
if "user_sessions" not in st.session_state:
    st.session_state.user_sessions = {}

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  

def sentiment_label(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def get_response(user_query, chat_history, bot_type, temperature, max_tokens, language="en"):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=None,
        max_retries=2,
    )

    if bot_type == "Chef":
        system_message = """You are a professional Chef AI, providing a range of recipes and cooking advice."""
    elif bot_type == "Teacher":
        system_message = """You are a knowledgeable Teacher AI with expertise in various subjects."""
    elif bot_type == "Nutritionist":
        system_message = """You are a professional Nutritionist AI."""
    elif bot_type == "Hr":
        system_message = """You are an HR consultant AI assisting freshers in job interview preparation."""
    elif bot_type == "Custom" and st.session_state.custom_system_message:
        system_message = st.session_state.custom_system_message

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{input}"),
        ]
    )

    user_query = f"Translate response to {language}. {user_query}"

    chain = prompt | llm | StrOutputParser()
    return chain.stream(
        {
            "history": chat_history,
            "input": user_query,
        }
    )

st.set_page_config(page_title="Multi-Chatbot App", page_icon="🤖")
st.title("Multi-Chatbot App")

with st.sidebar:
    st.header("Choose a Chatbot")
    bot_choice = st.selectbox("Select a bot:", ["Chef", "Teacher", "Nutritionist", "Hr", "Custom"])

    temperature = st.slider("Creativity Level (Temperature):", 0.0, 1.0, 0.5)
    max_tokens = st.slider("Response Length (Max Tokens):", 50, 1000, 256)

    language = st.selectbox("Select Response Language:", ["en", "es", "fr", "de"])

    custom_bot = st.checkbox("Custom Bot")
    if custom_bot:
        st.session_state.custom_system_message = st.text_area("Enter custom system message for the bot:")
        bot_choice = "Custom"

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")


    if st.button("Download Conversation (PDF)"):
        conversation = "\n".join([f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in st.session_state.chat_history])
        if conversation:
            pdf_file = "chat_history.pdf"
            c = canvas.Canvas(pdf_file, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica", 12)
            y = height - 40
            for line in conversation.split("\n"):
                c.drawString(40, y, line)
                y -= 20
                if y < 40:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = height - 40
            c.save()
            with open(pdf_file, "rb") as f:
                st.download_button(label="Download PDF", data=f, file_name=pdf_file, mime="application/pdf")
        else:
            st.warning("No conversation history available to create a PDF.")

    if st.button("Download Conversation (CSV)"):
        conversation_data = [{"User": msg.content if isinstance(msg, HumanMessage) else "", "Bot": msg.content if isinstance(msg, AIMessage) else ""} for msg in st.session_state.chat_history]
        df = pd.DataFrame(conversation_data)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name="chat_history.csv", mime="text/csv")


for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        sentiment = analyze_sentiment(message.content)
        with st.chat_message("Human"):
            st.markdown(f"{message.content} (Sentiment: {sentiment_label(sentiment)})")
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input("Your message")
if user_query:
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in get_response(user_query, st.session_state.chat_history, bot_choice, temperature, max_tokens, language):
            full_response += chunk
            message_placeholder.markdown(full_response)

        st.session_state.chat_history.append(AIMessage(full_response))

with st.expander("Chatbot Analytics Dashboard"):
    total_chats = len(st.session_state.chat_history)
    avg_sentiment = sum(analyze_sentiment(msg.content) for msg in st.session_state.chat_history if isinstance(msg, HumanMessage)) / max(len(st.session_state.chat_history), 1)
    st.write("Total Chat Sessions:", total_chats)
    st.write("Average Sentiment Score:", avg_sentiment)
