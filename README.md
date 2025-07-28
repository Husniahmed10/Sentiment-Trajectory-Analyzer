# 📈 Sentiment Trajectory Analyzer

**Sentiment Trajectory Analyzer** is an interactive web application that analyzes movie reviews and visualizes how sentiment evolves throughout the review — sentence by sentence. Instead of providing just a single verdict, this tool helps you trace the emotional arc of the text from beginning to end.

---

## 🔍 Overview

- Predicts overall **sentiment** (Positive/Negative) using a trained RNN model.
- Visualizes **sentiment progression** across the review to highlight emotional shifts.
- Offers a clean, dark-themed interface for interactive exploration.

---

## 🧰 Tools & Technologies Used

- **TensorFlow / Keras** – For building and training the RNN sentiment classification model.
- **Streamlit** – For creating the interactive web interface.
- **Pandas & NumPy** – For data processing and sentiment score tracking.
- **Regular Expressions (re)** – For splitting and parsing input text.
- **IMDB Movie Review Dataset** – Preprocessed binary sentiment dataset (built into Keras).

---

## 📊 Dataset

- **Source**: [Keras IMDB dataset](https://keras.io/api/datasets/imdb/)
- **Size**: 50,000 movie reviews (25k train / 25k test)
- **Labels**: Binary (0 = Negative, 1 = Positive)
- **Preprocessing**: Tokenized reviews, top 10,000 words used, sequences padded to fixed length.

---

## 🚀 Key Features

- Simple RNN-based classifier for sentiment detection.
- Sentence-level sentiment trajectory chart.
- Confidence score for final sentiment.
- Responsive and styled Streamlit UI.

---

## 📂 Output

- **Overall Sentiment** with confidence level.
- **Line chart** showing sentiment flow across the review.
- Helpful for understanding nuanced opinions or emotional transitions in text.
