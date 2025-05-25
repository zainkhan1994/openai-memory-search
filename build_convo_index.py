import json
import os
from datetime import datetime
from tqdm import tqdm

# Load metadata
METADATA_PATH = "flattened_output/zain_metadata.json"
THREAD_INDEX_PATH = "flattened_output/thread_index.json"

with open(METADATA_PATH, "r") as f:
    records = json.load(f)

# Index by date
threads_by_date = {}

for r in records:
    ts = r.get("timestamp") or r.get("create_time") or ""
    try:
        dt = datetime.fromisoformat(ts)
        date_str = dt.strftime("%Y-%m-%d")
    except:
        date_str = "unknown"

    if date_str not in threads_by_date:
        threads_by_date[date_str] = []

    threads_by_date[date_str].append({
        "role": r["role"],
        "content": r["content"]
    })

# Save the thread index
with open(THREAD_INDEX_PATH, "w") as f:
    json.dump(threads_by_date, f, indent=2)

print(f"✅ Indexed {len(records)} messages into {len(threads_by_date)} days.")

