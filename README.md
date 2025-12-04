# 🧠 ProTutor AI – AI Study Assistant

An offline AI Study Assistant that helps you learn better using:
✅ Summarizer (brief/detailed)
✅ Quiz generator (interactive with answers)
✅ RAG-based chatbot for document Q&A


# Project Creator:
AKASH DEEP SARKAR [24MTRDI01]

SNEHALATHA SH [24PHDDI12]


### 🚀 Features:
- Powered by Mistral 7B GGUF (local LLM)
- Built with Streamlit + Python
- Everything runs locally without internet
- RAG chatbot using FAISS + sentence-transformers


## 📦 Model Instructions

This app uses Mistral-7B GGUF model, which is not included in the repo.
Download it manually from:

👉 [https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

### Recommended file:
- `mistral-7b-instruct-v0.1.Q4_K_M.gguf`

Save the file to the `models/` directory like so:


### 📦 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
