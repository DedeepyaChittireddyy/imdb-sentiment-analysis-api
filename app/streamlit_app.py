import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

# --- Page Config ---
st.set_page_config(page_title="🎬 IMDb Sentiment Analyzer", layout="centered")

# --- Header ---
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 0.2em;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #666;
            margin-bottom: 1.5em;
        }
        .result {
            font-size: 1.3em;
            text-align: center;
            margin-top: 1em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎬 IMDb Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Instantly predict the sentiment of any movie review using BERT</div>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
Built with:
- 🤗 HuggingFace Transformers  
- 🐍 PyTorch  
- 🖼️ Streamlit

By **Dedeepya Chittireddy**
""")

# --- Load Model and Tokenizer ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- Examples ---
examples = [
    "Absolutely loved the film, everything was perfect!",
    "What a waste of time. Bad acting and terrible plot.",
    "The story was okay, not too good or bad."
]

with st.expander("💡 Try an example"):
    example_choice = st.radio("Choose a review example:", examples)
    if example_choice:
        st.session_state["example_text"] = example_choice
    else:
        st.session_state["example_text"] = ""

# --- Centered Input ---
col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    with st.form("sentiment_form"):
        user_input = st.text_area("📝 Enter your movie review:", value=st.session_state.get("example_text", ""), height=180, placeholder="Type or select an example...")
        submitted = st.form_submit_button("🚀 Analyze")

# --- Prediction Logic ---
if submitted:
    if not user_input.strip():
        st.error("❌ Please enter a review before submitting.")
    else:
        with st.spinner("🔍 Analyzing..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
                sentiment = "😊 Positive" if predicted.item() == 1 else "😞 Negative"

        # --- Display Results ---
        st.markdown("<div class='result'>🎯 <b>Sentiment:</b> {}</div>".format(sentiment), unsafe_allow_html=True)
        st.markdown("<div class='result'>📊 <b>Confidence:</b> {:.2%}</div>".format(confidence.item()), unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>📌 Built by Dedeepya Chittireddy using Transformers & Streamlit</p>", unsafe_allow_html=True)
