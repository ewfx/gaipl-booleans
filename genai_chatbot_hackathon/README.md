# GenAI Platform Support Chatbot 🚀

## Features
✅ Dry-Run Mode  
✅ Vector Search (Pinecone / Azure AI Search)  
✅ Real-time Streaming Capable  
✅ Approval Checkpoint for Risky Commands  
✅ Streamlit UI  

## Run
1. `pip install -r requirements.txt`
2. `python app.py`
3. `streamlit run streamlit_app.py`

## 🔐 Environment Setup
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-openai-key
OPENAI_API_BASE=https://api.openai.com/v1
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-env
```
Then, in your Python files:
```python
from dotenv import load_dotenv
load_dotenv()
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
```
