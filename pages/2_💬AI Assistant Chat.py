import streamlit as st
from utils.constants import info
import torch
from utils.model_loader import load_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

st.title("💬 Chat with My AI Assistant")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style/styles_chat.css")

# Get the variables from constants.py
pronoun = info['Pronoun']
name = info['Name']
subject = info['Subject']
full_name = info['Full_Name']

# Initialize the chat history
if "messages" not in st.session_state:
    welcome_msg = f"Hi! I'm {name}'s AI Assistant, Buddy. How may I assist you today?"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# App sidebar
with st.sidebar:
    st.markdown("""
                # Chat with my AI assistant
                """)
    with st.expander("Click here to see FAQs"):
        st.info(
            f"""
            - What are {pronoun} strengths and weaknesses?
            - What is {pronoun} expected salary?
            - What is {pronoun} latest project?
            - When can {subject} start to work?
            - Tell me about {pronoun} professional background
            - What is {pronoun} skillset?
            - What is {pronoun} contact?
            - What are {pronoun} achievements?
            """
        )

    messages = st.session_state.messages
    if messages is not None:
        st.download_button(
            label="Download Chat",
            data=json.dumps(messages),
            file_name='chat.json',
            mime='json',
        )

    st.caption(f"© Made by {full_name} 2023. All rights reserved.")

with st.spinner("Initiating the AI assistant. Please hold..."):
    # Check for GPU availability and set the appropriate device for computation.
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = "gpt2"  # Replace with your desired model

    # Load the tokenizer and model with retries
    tokenizer, model = load_model_and_tokenizer(model_name, DEVICE)

    # Function to generate text using the local model
    def generate_text(prompt, max_length=150):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the bio.txt file
with open("bio.txt", "r") as file:
    bio = file.read()

def ask_bot(user_query):
    prompt = f"You are an AI assistant for {name}. Here is some information about {name}:\n\n{bio}\n\nQuestion: {user_query}\nAnswer:"
    response = generate_text(prompt)
    return response.strip()

# After the user enters a message, append that message to the message history
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the chat history with unique keys to avoid DuplicateWidgetID error
for index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.text_area(f"You_{index}: ", value=message["content"], disabled=True, key=f"user_{index}")
    elif message["role"] == "assistant":
        st.text_area(f"Assistant_{index}: ", value=message["content"], disabled=True, key=f"assistant_{index}")

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant" and prompt:
    with st.spinner("🤔 Thinking..."):
        response = ask_bot(prompt)
        # Check if the response is redundant (based on previous responses)
        previous_responses = [msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"]
        if response in previous_responses:
            response = generate_text(prompt)  # Generate a new response if the current one is redundant
        st.session_state.messages.append({"role": "assistant", "content": response})

# Suggested questions
questions = [
    f'What are {pronoun} strengths and weaknesses?',
    f'What is {pronoun} latest project?',
    f'When can {subject} start to work?'
]

def send_button_ques(question):
    st.session_state.disabled = True
    response = ask_bot(question)
    st.session_state.messages.append({"role": "user", "content": question}) # display the user's message first
    st.session_state.messages.append({"role": "assistant", "content": response}) # display the AI message afterwards
    
if 'button_question' not in st.session_state:
    st.session_state['button_question'] = ""
if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False
    
if st.session_state['disabled'] == False: 
    for n, msg in enumerate(st.session_state.messages):
        # Render suggested question buttons
        buttons = st.container()
        if n == 0:
            for q in questions:
                button_ques = buttons.button(label=q, on_click=send_button_ques, args=[q], disabled=st.session_state.disabled)
