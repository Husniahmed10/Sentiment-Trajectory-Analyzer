# Step 1: Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re 

st.set_page_config(
    page_title="Sentiment Trajectory Analyzer",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        body { font-family: 'Roboto', sans-serif; }
        .stApp { background-color: #0F172A; } /* Dark Navy background */

        /* Title with a new, professional gradient */
        .main-title {
            font-size: 48px; font-weight: 700; text-align: center; margin-bottom: 20px;
            background: -webkit-linear-gradient(45deg, #2563EB, #4F46E5); /* Blue to Indigo gradient */
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subheader { font-size: 18px; color: #CBD5E0; text-align: center; margin-bottom: 2rem; }
        
        /* Main Result Card */
        .result-card {
            background: #1E293B; border-radius: 15px; border-left: 7px solid;
            padding: 1.5rem 2rem; margin: 1rem 0 2rem 0;
        }
        .positive-border { border-left-color: #22C55E; } /* Green */
        .negative-border { border-left-color: #EF4444; } /* Red */
        
        .verdict-header { font-size: 16px; color: #94A3B8; font-weight: bold; margin-bottom: 0.5rem;}
        .verdict-value { font-size: 28px; font-weight: 700; }
        .positive-text { color: #22C55E; } .negative-text { color: #EF4444; }
        
        /* Trajectory Analysis Section */
        .analysis-header {
            font-size: 24px; font-weight: bold; color: #FFFFFF; text-align: center;
            margin-top: 3rem; margin-bottom: 1rem; border-bottom: 2px solid #334155; padding-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_index():
    model = load_model('simple_rnn_imdb.h5')
    word_index = imdb.get_word_index()
    return model, word_index

model, word_index = load_model_and_index()
VOCAB_SIZE = 10000

def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words if word_index.get(w, 2) < (VOCAB_SIZE - 3)]
    return sequence.pad_sequences([encoded], maxlen=500)

def analyze_sentiment_trajectory(text, model):
    """
    Analyzes the sentiment of the text cumulatively, sentence by sentence.
    Returns a DataFrame suitable for plotting.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return pd.DataFrame()
        
    scores = []
    cumulative_text = ""
    for i, sentence in enumerate(sentences):
        cumulative_text += " " + sentence
        preprocessed = preprocess_text(cumulative_text)
        score = model.predict(preprocessed, verbose=0)[0][0]
        scores.append(score)

    # Create a DataFrame for plotting
    chart_data = pd.DataFrame({
        'Progression': range(1, len(scores) + 1),
        'Sentiment Score': scores,
    }).set_index('Progression')

    return chart_data

st.markdown('<div class="main-title">Sentiment Trajectory Analyzer</div>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Go beyond the final verdict. Visualize the emotional arc of any review.</p>', unsafe_allow_html=True)

user_input = st.text_area(
    "Enter a movie review:",
    height=150,
    placeholder="e.g., At first, I was skeptical, the beginning felt slow. But then, the plot exploded into a masterpiece of incredible storytelling that left me absolutely thrilled.",
    label_visibility="collapsed"
)

_, btn_col, _ = st.columns([1, 1.2, 1])
analyze_button = btn_col.button("📈 Perform Deep Analysis", use_container_width=True)

if analyze_button and user_input.strip() != "":
    with st.spinner("Conducting deep structural analysis..."):
        final_prediction = model.predict(preprocess_text(user_input), verbose=0)[0][0]
        final_sentiment = "Positive" if final_prediction > 0.5 else "Negative"
        sentiment_class = "positive" if final_sentiment == "Positive" else "negative"
        
        trajectory_data = analyze_sentiment_trajectory(user_input, model)

    st.markdown(f"""
        <div class="result-card {sentiment_class}-border">
            <div class="verdict-header">OVERALL VERDICT</div>
            <span class="verdict-value {sentiment_class}-text">{final_sentiment}</span>
            <span style="color: #94A3B8; font-size: 20px; margin-left: 10px;">(Confidence: {final_prediction:.1%})</span>
        </div>
    """, unsafe_allow_html=True)
    
    if not trajectory_data.empty:
        st.markdown('<div class="analysis-header">Sentiment Trajectory</div>', unsafe_allow_html=True)
        st.write("This chart shows how the sentiment evolved as the review progressed. It reveals the emotional journey from the first sentence to the last.")
        
        st.line_chart(trajectory_data, color="#3B82F6") 
        
        st.caption("Each point on the X-axis represents one additional sentence from the review.")
    else:
        st.warning("Could not generate a trajectory. Please enter a review with at least one full sentence.")

else:
    st.info("The Analysis Engine is ready. Enter a review to chart its emotional journey.")


st.markdown("---")
st.caption("A TensorFlow RNN application featuring Sentiment Trajectory Visualization.")