import streamlit as st
import torch
from utils.constants import info
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
    def generate_text(prompt, max_length=50):
    # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")

    # Ensure input_ids length does not exceed model's maximum sequence length
        max_input_length = tokenizer.model_max_length
        if inputs.input_ids.size(1) > max_input_length:
            inputs.input_ids = inputs.input_ids[:, :max_input_length]

    # Move inputs to appropriate device
        inputs = inputs.to(DEVICE)

    # Generate text with the model
        outputs = model.generate(inputs.input_ids, max_length=max_length)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Load the bio.txt file
    with open("bio.txt", "r") as file:
        bio = file.read()

def ask_bot(user_query):
    prompt = f"You are an AI assistant for {name}. Here is some information about {name}:\n\n{bio}\n\n{user_query}"
    response = generate_text(prompt)
    return response

# Handling user input and message display
user_query = st.text_input("Your question")
if st.button("Send"):
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        response = ask_bot(user_query)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Displaying messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.text_area("You:", value=message["content"], disabled=True)
        elif message["role"] == "assistant":
            st.text_area("Assistant:", value=message["content"], disabled=True)

# Suggested questions
questions = [
    f'What are {pronoun} strengths and weaknesses?',
    f'What is {pronoun} latest project?',
    f'When can {subject} start to work?'
]

# Function to handle button click
def send_button_ques(question):
    response = ask_bot(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": response})

# Render suggested question buttons
for q in questions:
    if st.button(q, on_click=send_button_ques, args=(q,)):
        pass  # Button will trigger `send_button_ques` function

