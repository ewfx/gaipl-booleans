
# 🚀 GenAI KB Chatbot - Deployment Guide

## 📂 Project Structure
```
genai_chatbot_hackathon/
├── app.py                  # Flask backend (Chatbot logic)
├── streamlit_app.py        # Streamlit frontend (Optional)
├── pinecone_sample.py      # Vector upload to Pinecone
├── kb_articles.json        # Sample KB articles
├── requirements.txt        # Python dependencies
├── .env                    # API keys and secrets
└── README.md               # Project overview
```

---

## ⚙️ Step 1: Install Requirements
Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔑 Step 2: Configure `.env`
Create a `.env` file in the root folder:
```
OPENAI_API_KEY=sk-your-openai-key
OPENAI_API_BASE=https://api.openai.com/v1
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
```

---

## 📥 Step 3: Populate Pinecone Vector Index
Run this to upload KB articles as vector embeddings:
```bash
python pinecone_sample.py
```
✅ Output: `✅ Pinecone index populated`

---

## 🟢 Step 4: Start Flask Backend API
```bash
python app.py
```
API is available at:
```
http://localhost:5000/chat
```

✅ Example Test Call:
```bash
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"issue": "Disk usage high", "dry_run": true}'
```

---

## 🌐 Step 5 (Optional): Launch Streamlit Frontend
```bash
streamlit run streamlit_app.py
```
Accessible at: `http://localhost:8501`

---

## 🚀 Step 6 (Optional): Dockerize for Deployment
### Sample Dockerfile
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```
Build & Run Docker:
```bash
docker build -t genai-chatbot .
docker run -p 5000:5000 genai-chatbot
```

---

## ✅ Hackathon Demo Checklist
| Feature                             | ✅ Status |
|-------------------------------------|----------|
| Dry-Run / Real Execution            | ✅ Ready |
| Vector Search (Pinecone)            | ✅ Ready |
| Risky Command Approval              | ✅ Ready |
| Real-time Streaming (Streamlit)     | ✅ Ready |
| OpenAI LLM                          | ✅ Ready |
| .env based secret management        | ✅ Ready |

---

## 📢 Pro Tips for Hackathon Demo
- Emphasize `.env` based secure key handling
- Demo Dry-run mode before real execution
- Showcase Pinecone vector search vs plain keyword search
- Use Streamlit as a user-friendly interface
- Explain risky command approval checkpoint (`rm -rf`, `shutdown`, etc.)

---


