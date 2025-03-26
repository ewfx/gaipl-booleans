from flask import Flask, request, jsonify
import json
import os
from dotenv import load_dotenv
import openai
import pinecone

# ✅ Load environment variables
load_dotenv()

# ✅ Azure OpenAI Configuration
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # ✅ Use correct API base
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")  # ✅ Use correct API key
openai.api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")  # ✅ Default if not set

AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")  # ✅ Correct deployment name

# ✅ Pinecone Initialization
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_client = pinecone.Pinecone(api_key=pinecone_api_key)

index_name = "kb-index"  # ✅ Ensure this matches your Pinecone index
index = pinecone_client.Index(index_name)  # ✅ Connect to Pinecone index

# ✅ Load KB Articles (from JSON file)
with open("kb_articles.json", encoding="utf-8") as f:
    kb_articles = json.load(f)

# ✅ Function to Generate Embeddings with Azure OpenAI
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model=AZURE_DEPLOYMENT_NAME  # ✅ Correct deployment name
    )
    return response.data[0].embedding  # ✅ Correct parsing


# ✅ Flask API Initialization
app = Flask(__name__)

# ✅ Chat Endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    issue = data.get("issue")
    dry_run = data.get("dry_run", True)

    # ✅ Generate embedding for issue
    embedding = get_embedding(issue)

    # ✅ Query Pinecone
    result = index.query(
        vector=embedding,
        top_k=1,
        include_metadata=True
    )

    print("Pinecone Query Result:", result)  # ✅ Debug output

    if not result["matches"]:  # ✅ Ensure correct attribute reference
        return jsonify({"error": "No KB match found"}), 404

    # ✅ Extract KB and Commands
    kb_used = result["matches"][0]["metadata"]["kb_article"]
    commands = [line for line in kb_used.split(".") if "`" in line]
    shell_commands = [line.split("`")[1] for line in commands if "`" in line]

    return jsonify({
        "kb_used": kb_used,
        "commands": "\n".join(shell_commands),
    })


# ✅ Run Flask App
if __name__ == "__main__":
    app.run(port=5000)
print("Response:", jsonify({...}))

