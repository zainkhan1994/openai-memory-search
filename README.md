# 🧠 OpenAI Semantic Memory CLI

Search your own exported ChatGPT or message data semantically — like asking your brain questions and getting smart results back.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Embeddings](https://img.shields.io/badge/Model-text--embedding--3--small-green)
![VectorDB](https://img.shields.io/badge/FAISS-enabled-purple)

---

## 📦 Features

- 🔍 Ask natural language questions about your own data
- 🧠 Embeds messages using `text-embedding-3-small`
- ⚡ Blazing-fast similarity search with FAISS
- ✅ Works offline once embedded
- 🛠 Easily extendable with filters or thread recovery

---

## 🧰 Setup

```bash
git clone https://github.com/zainkhan1994/openai-memory-search.git
cd openai-memory-search
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
🔐 API Key
Create a .env file:

ini
Copy
Edit
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
📂 Input File Required
Place your flattened .jsonl from exported conversations here:

bash
Copy
Edit
flattened_output/conversations.jsonl
You can flatten OpenAI exports using embed_and_index.py or any tool that outputs {"content": "...", "role": "..."} per line.

🚀 Embed Your Memory
bash
Copy
Edit
python embed_and_index.py
Generates zain_index.faiss + zain_metadata.json

Skips oversized or invalid messages automatically

🧠 Query Your Mind
bash
Copy
Edit
python semantic_search.py
Example queries:

when did I talk about Pinecone?

netSuite vs Oracle

personal AI assistant setup

🧪 Output Preview
yaml
Copy
Edit
🔍 Ask your brain: epm and forecasting

🧠 Top Results:

[11948] Score: 0.80
→ Role: user
→ Content:
What are some competitors in the EPM space?
--------------------------------------------------
[13806] Score: 0.85
→ Role: user
→ Content:
Do a graphic showing EPM services
--------------------------------------------------
🔮 Next Up (optional features)
 Date filters

 Role filters (user vs assistant)

 Thread view mode

 Markdown export

 GUI w/ Streamlit or TUI

🧠 Built by Zain Khan
A personal memory interface — from exported chat to semantically searchable archive.
