import time
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_name, device, retries=3, delay=5):
    for attempt in range(retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            return tokenizer, model
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    st.error("Failed to load model or tokenizer after multiple attempts.")
    st.stop()
