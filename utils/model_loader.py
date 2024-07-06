# File: utils/model_loader.py

import time
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name, device, retries=3, delay=5):
    for attempt in range(retries):
        try:
            st.info(f"Attempt {attempt + 1}: Loading model and tokenizer for '{model_name}'")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            st.success("Model and tokenizer loaded successfully.")
            return tokenizer, model
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    st.error("Failed to load model or tokenizer after multiple attempts.")
    st.stop()
