# === GPT-2 ONLY VERSION of HopeBot Chatbot ===

import streamlit as st
import pandas as pd
import torch
import time
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import textwrap
import re
import html

# === Page config ===
st.set_page_config(page_title="HopeBot Chatbot", layout="centered")

# === Unicode fix ===
def safe_text(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

# === Custom CSS ===
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    div.block-container {
        background-color: #ffffff !important;
    }
    header[data-testid="stHeader"],
    section[data-testid="stChatInput"],
    main, footer, .css-uf99v8, .css-1y4p8pa {
        background-color: #ffffff !important;
    }
    section[data-testid="stChatInput"] {
        border-top: 1px solid #ccc;
    }
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #ffffff !important;
        box-shadow: 4px 0 10px rgba(0, 0, 0, 0.3);
        border-right: 1px solid #333;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    div[data-testid="stMarkdown"] > div {
        font-size: 16px;
        line-height: 1.6;
    }
    div[data-testid="stMarkdown"]:has(div:contains('🤖')) {
        background-color: #52b788;
        padding: 10px 15px;
        border-radius: 15px;
        margin-right: auto;
        max-width: 70%;
        margin-bottom: 10px;
        color: #000000;
    }
    div[data-testid="stMarkdown"]:has(div:contains('🙋‍♀️')) {
        background-color: #ffffff;
        padding: 10px 15px;
        border-radius: 15px;
        margin-left: auto;
        max-width: 70%;
        margin-bottom: 10px;
        border: 1px solid #ccc;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# === Load GPT-2 model ===
tokenizer = GPT2Tokenizer.from_pretrained("Lisarahaman13/hopebot-exp-10", use_auth_token=st.secrets["HF_TOKEN"])
gpt2_model = GPT2LMHeadModel.from_pretrained("Lisarahaman13/hopebot-exp-10", use_auth_token=st.secrets["HF_TOKEN"])

# === Cleaning function ===
def clean_response(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = html.unescape(text)
    text = text.replace("�", "'").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("–", "-")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\.\s*\.+", ".", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    text = re.sub(r"I\.\s*$", "", text)
    text = re.sub(r"don'\s*t", "don't", text, flags=re.IGNORECASE)
    text = re.sub(r"n'\s*t", "n't", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\x00-\x7F]+", "'", text)

    if len(text.split()) > 50:
        text = " ".join(text.split()[:45]) + "..."
    if text.endswith(("I'm", "I", "you", "but", "and", "to", "too")):
        text = text.rsplit(" ", 1)[0] + "."
    if not text.endswith(('.', '?', '!')):
        text += '.'

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 2:
        text = " ".join(sentences[:2])

    return text.strip()

# === Generate GPT-2 response ===
def generate_gpt2_response(user_input):
    prompt = f"User: {user_input}\nHopeBot:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_k=30,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split(prompt)[-1].strip() if prompt in decoded else decoded.strip()
    response = clean_response(response)
    return response

# === Init chat history ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Sidebar ===
with st.sidebar:
    st.markdown("### 🧠 HopeBot Mental Health Assistant")
    st.markdown("Developed by: Lisa Rahaman")
    st.markdown("---")
    st.markdown("💬 About HopeBot")
    st.markdown("HopeBot is your AI companion designed to support your emotional well-being.")
    st.markdown("You can talk to it about stress, anxiety, sleep, and more — anytime, anywhere.")
    st.markdown("---")
    st.markdown("📌 Disclaimer")
    st.markdown("HopeBot is not a licensed therapist.")
    st.markdown("It provides general support and coping strategies.")
    st.markdown("For serious concerns, please consult a mental health professional.")
    st.markdown("---")

# === Title ===
st.title("🤖 HopeBot - Mental Health Chatbot")
st.markdown("Hi! I am HopeBot, your companion. Reminder, I am just a companion, not a replacement of your therapist. You still need to consult with your therapist <3")
st.markdown("You can ask anything about your feelings, stress, panic, sleep, etc.")

# === Display chat history ===
for chat in st.session_state.chat_history:
    timestamp = chat.get("timestamp", "")
    if chat["role"] == "user":
        st.markdown(f"""
            <div style='background-color:#f0f0f0; padding:10px 15px; border-radius:15px; 
                        max-width:70%; margin-left:auto; margin-bottom:10px; color:#000000;
                        font-size:16px; border: 1px solid #cccccc;'>
                🙋‍♀️You: {safe_text(chat["message"])}
                <div style='font-size:12px; color: #666; text-align: right;'>{timestamp}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color:#52b788; padding:10px 15px; border-radius:15px; 
                        max-width:70%; margin-right:auto; margin-bottom:10px; color:#000000;
                        font-size:16px;'>
                🤖HopeBot: {safe_text(chat["message"])}
                <div style='font-size:12px; color: #003300; text-align: left;'>{timestamp}</div>
            </div>
        """, unsafe_allow_html=True)

# === Chat input ===
user_input = st.chat_input("Type your question here:")

if user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "message": user_input,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    start_time = time.time()
    reply = generate_gpt2_response(user_input)
    end_time = time.time()
    response_time = round(end_time - start_time, 2)
    reply += f"\n\n⏱️ Response time: {response_time} seconds"

    with open("chat_log.txt", "a", encoding="utf-8", errors="ignore") as f:
        f.write(f"{datetime.now()} | User: {safe_text(user_input)}\n")
        f.write(f"{datetime.now()} | HopeBot: {safe_text(reply)}\n\n")

    st.session_state.chat_history.append({
        "role": "assistant",
        "message": reply,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    st.rerun()