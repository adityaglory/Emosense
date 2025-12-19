# 🤯 EmoSense 3.0: Multi-Label Emotion AI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![Model](https://img.shields.io/badge/AI-DistilBERT-orange)
![License](https://img.shields.io/badge/License-MIT-green)

<img width="1900" height="800" alt="Screenshot 2025-12-19 153738" src="https://github.com/user-attachments/assets/e73a80a5-df16-404c-b9f3-08cce1600db9" />

**EmoSense 3.0** is a production-grade NLP Microservice capable of detecting **28 distinct emotional states** (e.g., *Admiration, Remorse, Bittersweet*) from a single sentence text input.

Unlike traditional sentiment analysis (Positive/Negative), EmoSense understands nuance. It uses a **Multi-Label Classification** head, allowing it to detect mixed feelings simultaneously (e.g., "I'm tired but happy" -> `Fatigue` + `Joy`).

## 🚀 Key Features

* **🧠 Advanced NLP:** Fine-tuned **DistilBERT** on the Google **GoEmotions** dataset (40k+ samples).
* **⚡ High Performance:** Model compressed via Dynamic Quantization (INT8) and exported to ONNX Runtime for 3x faster inference CPU.
* **🐳 Dockerized:** Fully containerized application for "Run Anywhere" deployment.
* **📊 Dashboard:** Interactive Streamlit UI with historical logging (SQLite).
* **📈 Multi-Label:** Sigmoid activation layer to handle complex/mixed emotions.

## 🛠️ Tech Stack

* **Training:** HuggingFace Transformers, PyTorch, Scikit-learn
* **Inference:** ONNX Runtime (CPU Optimized)
* **Backend/Frontend:** Streamlit, SQLAlchemy (SQLite)
* **DevOps:** Docker, Git

## 💻 How to Run (Docker)

The easiest way to run EmoSense is via Docker. No environment setup required.

# 1. Clone this repository
```bash
git clone [https://github.com/adityaglory/Emosense.git](https://github.com/adityaglory/Emosense.git)
cd Emosense
```
# 2. Build the image
```bash
docker build -t emosense-v3 .
```

# 3. Run container
```bash
docker run -p 8501:8501 emosense-v3
```
Access the app at: http://localhost:8501

📂 Project Structure

├── artifacts/          # Quantized ONNX Model (INT8) & Tokenizer
├── src/                # Source Code
│   ├── app.py          # Streamlit Dashboard (Frontend)
│   ├── train.py        # Fine-Tuning Script (Multi-Label Trainer)
│   ├── convert.py      # ONNX Export & Quantization
│   └── database.py     # SQLite Database Handler
├── Dockerfile          # Container Configuration
└── requirements.txt    # Python Dependencies

