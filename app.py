import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
MODEL_PATH = "model"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.to(device)
model.eval()

# Label mapping
label_map = {0: "🧠 Human", 1: "🤖 AI-generated"}

# Streamlit UI
st.set_page_config(page_title="AI Text Detector", layout="centered")
st.title("🕵️ AI vs Human Text Classifier")

st.markdown("Paste any text below and I'll tell you whether it's **AI-generated** or **human-written**.")

text_input = st.text_area("✍️ Enter your text here:", height=200)

if st.button("Detect"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        # Tokenize and move to device
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0][pred].item()

        st.success(f"**Prediction:** {label_map[pred]}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
