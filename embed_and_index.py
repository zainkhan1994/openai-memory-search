import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tiktoken import get_encoding

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Constants
EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS = 8192
DATA_PATH = "flattened_output/conversations.jsonl"
INDEX_PATH = "flattened_output/zain_index.faiss"
METADATA_PATH = "flattened_output/zain_metadata.json"

# Load tokenizer
enc = get_encoding("cl100k_base")

# Load records
with open(DATA_PATH, "r") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Sanitize + filter
texts = []
valid_records = []
for r in records:
    content = r.get("content", "")
    if not isinstance(content, str):
        continue
    stripped = content.strip()
    if not stripped or stripped in {"{}", "[]"} or len(stripped) < 3:
        continue
    token_len = len(enc.encode(stripped))
    if token_len < MAX_TOKENS:
        texts.append(stripped)
        valid_records.append(r)

print(f"🧠 {len(valid_records)} records ready for embedding")

# Embed in batches
embeddings = []
BATCH_SIZE = 100
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    try:
        res = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeds = [e.embedding for e in res.data]
        embeddings.extend(batch_embeds)
        print(f"  → Embedded batch {i // BATCH_SIZE + 1}")
    except Exception as e:
        print(f"❌ Batch {i // BATCH_SIZE + 1} failed: {e}")
        continue

# Save FAISS index
if embeddings:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index saved: {INDEX_PATH}")

    with open(METADATA_PATH, "w") as f:
        json.dump(valid_records, f)
    print(f"✅ Metadata saved: {METADATA_PATH}")
else:
    print("⛔ No embeddings created. Check previous batch errors.")

