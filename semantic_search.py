import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# Setup
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load metadata and index
with open("flattened_output/zain_metadata.json", "r") as f:
    metadata = json.load(f)

index = faiss.read_index("flattened_output/zain_index.faiss")

# Build CLI loop
def query_loop():
    while True:
        query = input("\n🔍 Ask your brain: ").strip()
        if not query:
            print("⛔ Empty query. Try again.")
            continue
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye.")
            break

        # Embed query
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            )
            query_vector = np.array(response.data[0].embedding).astype("float32")
        except Exception as e:
            print(f"❌ Error embedding query: {e}")
            continue

        # Search FAISS
        D, I = index.search(np.array([query_vector]), k=5)
        print("\n🧠 Top Results:\n")
        for idx, score in zip(I[0], D[0]):
            if idx >= len(metadata):
                continue
            msg = metadata[idx]
            print(f"[{idx}] Score: {score:.2f}")
            print(f"→ Role: {msg['role']}")
            print(f"→ Content:\n{msg['content'][:1000]}")
            print("-" * 50)

# Start it
if __name__ == "__main__":
    query_loop()

