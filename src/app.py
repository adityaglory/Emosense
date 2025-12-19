import streamlit as st
import onnxruntime as ort
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from database import init_db, log_prediction, engine
from sqlalchemy import text
from typing import Tuple

st.set_page_config(page_title="EmoSense 3.0", page_icon="🤯", layout="wide")
init_db()

@st.cache_resource
def load_resources() -> Tuple[AutoTokenizer, ort.InferenceSession]:
    try:
        tokenizer = AutoTokenizer.from_pretrained("./artifacts/model")
        # Load model Quantized
        session = ort.InferenceSession(
            "./artifacts/model_quant.onnx", 
            providers=["CPUExecutionProvider"]
        )
        return tokenizer, session
    except Exception as e:
        return None, None

tokenizer, session = load_resources()

# 28 LABELS GOEMOTIONS
LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 GoEmotions Dashboard")
    st.caption("Multi-Label Detection Active")
    
    st.subheader("Recent Activity")
    with engine.connect() as conn:
        query = text("SELECT predicted_label, confidence_score, input_text FROM prediction_logs ORDER BY id DESC LIMIT 5")
        df_recent = pd.read_sql(query, conn)
    
    if not df_recent.empty:
        for _, row in df_recent.iterrows():
            st.markdown(f"**{row['predicted_label']}**")
            st.caption(f"Conf: {row['confidence_score']:.0%} | '{row['input_text'][:20]}...'")
            st.divider()
    
    if st.button("🔄 Refresh Data"): st.rerun()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- MAIN APP ---
if not tokenizer or not session:
    st.error("❌ Model not found! Wait for training to finish & run convert.py.")
    st.stop()

st.title("🤯 EmoSense 3.0: Multi-Label AI")
st.markdown("### Powered by DistilBERT (28 Emotions)")
st.caption("This model can detect **mixed feelings** (e.g., Joy + Relief + Excitement) simultaneously.")

col1, col2 = st.columns([2, 1])

with col1:
    with st.form("prediction_form"):
        user_text = st.text_area("How are you feeling?", height=150, placeholder="I finally finished the project, I'm so tired but happy!")
        submitted = st.form_submit_button("Analyze Complexity ⚡", type="primary")

if submitted and user_text:
    inputs = tokenizer(user_text, return_tensors="np", padding=True, truncation=True)
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    logits = session.run(None, ort_inputs)[0]
    probs = sigmoid(logits[0])
    top_indices = np.argsort(probs)[-5:][::-1]
    
    primary_label = LABELS[top_indices[0]]
    primary_conf = float(probs[top_indices[0]])
    
    log_prediction(user_text, primary_label, primary_conf, version="v3.0-MultiLabel")
    
    with col2:
        st.success("Analysis Complete")
        
        found_emotion = False
        for idx in top_indices:
            score = float(probs[idx])
            label = LABELS[idx]
            
            if score > 0.10: # Threshold 10%
                found_emotion = True
                st.metric(label.upper(), f"{score:.1%}")
                st.progress(score)
        
        if not found_emotion:
            st.info("No strong emotion detected (Neutral).")

    st.subheader("Emotion Spectrum")
    top_10_indices = np.argsort(probs)[-10:]
    chart_data = pd.DataFrame({
        "Emotion": [LABELS[i] for i in top_10_indices], 
        "Confidence": probs[top_10_indices]
    })
    st.bar_chart(chart_data.set_index("Emotion"), color="#FF4B4B")