import os
import json
from dotenv import load_dotenv
import openai
import pinecone
from pinecone import ServerlessSpec  # ✅ Required for creating Pinecone index

# Load environment variables
load_dotenv()

# ✅ Corrected Azure OpenAI Configuration
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # ✅ Load from .env
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")  # ✅ Load from .env
openai.api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")  # ✅ Load from .env or default

AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")  # ✅ Corrected Deployment Name

# ✅ Corrected Pinecone Initialization
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # ✅ Load from .env
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")  # ✅ Load from .env

pinecone_client = pinecone.Pinecone(api_key=pinecone_api_key)

# ✅ Check if Pinecone index exists, create if necessary
index_name = "kb-index"
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=1536,  # ✅ Match embedding model dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ Required spec argument
    )

# ✅ Connect to Pinecone index
index = pinecone_client.Index(index_name)

# ✅ Function to Generate Embeddings with Azure OpenAI
def get_embedding(text):
    response = openai.embeddings.create(  # ✅ Corrected function call for latest OpenAI SDK
        input=text,
        model=AZURE_DEPLOYMENT_NAME  # ✅ Correct usage of deployment name
    )
    return response.data[0].embedding  # ✅ Corrected response parsing


# ✅ Load KB articles
with open("kb_articles.json", encoding="utf-8") as f:
    articles = json.load(f)

# ✅ Create embeddings and upload to Pinecone
vector_data = []
for article in articles:
    embedding = get_embedding(article["content"])
    vector_data.append((article["id"], embedding, {"kb_article": article["content"]}))

index.upsert(vectors=vector_data)

print("✅ Pinecone Index Updated with Azure OpenAI Embeddings!")
print(index.describe_index_stats())

