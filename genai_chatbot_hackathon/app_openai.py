from flask import Flask, request, jsonify
import json
import subprocess
import pinecone
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# Initialize Flask
app = Flask(__name__)

# Load KB articles (can also be replaced with DB or file read)
with open('kb_articles.json') as f:
    kb_articles = json.load(f)

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pinecone.Index('your-kb-index')

# Risky command checker
def risky(command):
    risky_keywords = ['rm -rf', 'shutdown', 'mkfs', 'reboot']
    return any(r in command for r in risky_keywords)

# Execute shell commands
def execute(commands, dry_run=True):
    log = []
    for cmd in commands:
        if risky(cmd):
            log.append(f"Approval Required: {cmd}")
            continue
        if dry_run:
            log.append(f"[Dry-Run] {cmd}")
        else:
            try:
                output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
                log.append(f"{cmd}\n{output}")
            except subprocess.CalledProcessError as e:
                log.append(f"{cmd}\n{e.output}")
    return "\n".join(log)

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    issue = data.get('issue')
    dry_run = data.get('dry_run', True)

    # Get embedding from OpenAI
    embedding = openai.Embedding.create(input=issue, model="text-embedding-ada-002")['data'][0]['embedding']
    
    # Vector search in Pinecone
    result = index.query(vector=embedding, top_k=1, include_metadata=True)
    if not result.matches:
        return jsonify({"error": "No KB match found"}), 404

    # Extract KB and commands
    kb_used = result.matches[0].metadata['kb_article']
    commands = [line for line in kb_used.split('.') if '`' in line]
    shell_commands = [line.split('`')[1] for line in commands if '`' in line]

    # Execute commands
    execution_result = execute(shell_commands, dry_run)
    return jsonify({
        "kb_used": kb_used,
        "commands": "\n".join(shell_commands),
        "execution_result": execution_result
    })

if __name__ == '__main__':
    app.run(port=5000)
