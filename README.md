# 🧠 OpenAI Memory Search - Enhanced

Transform your ChatGPT conversations into a powerful, searchable knowledge base with mind maps, notes, and analytics.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Embeddings](https://img.shields.io/badge/Model-text--embedding--3--small-green)
![VectorDB](https://img.shields.io/badge/FAISS-enabled-purple)
![UI](https://img.shields.io/badge/Streamlit-Web_Interface-red)

---

## ✨ Enhanced Features

### 🔍 **Semantic Search**
- Natural language queries about your conversations
- Advanced filters by role, date, and topic
- Real-time search results with conversation context
- Export conversations to text files

### 🗺️ **Interactive Mind Maps**
- Visual network graph of conversation topics
- Topic clustering with keyword analysis
- Interactive exploration of related conversations
- Downloadable visualizations

### 📝 **Notes & Annotations**
- Add personal notes to any conversation
- Persistent note storage with conversation linking
- Notes manager for easy organization
- Markdown support for rich text notes

### 📊 **Analytics Dashboard**
- Conversation statistics and metrics
- Timeline visualization of your chat history
- Role distribution analysis
- Topic trends and patterns

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/zainkhan1994/openai-memory-search.git
cd openai-memory-search
pip install -r requirements.txt
```

### 2. Setup API Key
Create a `.env` file:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. Prepare Your Data
Place your conversation data in `flattened_output/conversations.jsonl`:
```jsonl
{"content": "Your message here", "role": "user", "conversation_id": "conv_001", "message_id": "msg_001", "timestamp": "1703000000"}
{"content": "AI response here", "role": "assistant", "conversation_id": "conv_001", "message_id": "msg_002", "timestamp": "1703000060"}
```

### 4. Build Index
```bash
python embed_and_index.py
```

### 5. Launch Enhanced Web Interface
```bash
streamlit run enhanced_app.py
```

---

## 🖥️ Interface Modes

### Search Mode
![Search Interface](https://github.com/user-attachments/assets/ef260a34-64dc-43db-9e84-cb1ff007dc7c)
- Type natural language queries
- Filter by role, date, and result count
- View full conversations with context

### Mind Map Mode
![Mind Map Interface](https://github.com/user-attachments/assets/6da5944c-e3c0-478c-8563-2f700bf6e6c8)
- Interactive network visualization
- Topic clustering and keyword analysis
- Explore conversation relationships

### Analytics Mode
![Analytics Dashboard](https://github.com/user-attachments/assets/eb91ff7b-9a3b-4dc0-8d19-2ab5a157e22d)
- Conversation metrics and statistics
- Timeline charts and role distributions
- Usage patterns and trends

---

## 🛠️ CLI Tools (Legacy)

For command-line usage:
```bash
# Simple CLI search
python semantic_search.py

# Build conversation index
python build_convo_index.py

# Batch process insights
python batch_process_insights.py
```

---

## 📁 Project Structure

```
openai-memory-search/
├── enhanced_app.py          # Main Streamlit web interface
├── streamlit_app.py         # Original Streamlit app
├── semantic_search.py       # CLI search tool
├── embed_and_index.py       # Data processing and indexing
├── flattened_output/        # Data storage directory
│   ├── conversations.jsonl  # Input conversation data
│   ├── zain_index.faiss    # Vector index
│   ├── zain_metadata.json  # Conversation metadata
│   └── conversation_notes.json # User notes
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

---

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings
- Optional: Configure models and paths in script headers

### Data Format
Input JSONL should contain:
- `content`: Message text
- `role`: "user" or "assistant"
- `conversation_id`: Unique conversation identifier
- `message_id`: Unique message identifier
- `timestamp`: Unix timestamp or ISO format

---

## 🎯 Use Cases

- **Research**: Search through months of AI conversations
- **Learning**: Visualize knowledge connections in mind maps
- **Organization**: Categorize and annotate important discussions
- **Analysis**: Track conversation patterns and topics over time
- **Export**: Save important conversations for reference

---

## 🚧 Advanced Features

### Custom Topic Clustering
Modify the `cluster_conversations()` function to add your own topic categories and keywords.

### Note Templates
Extend the notes system with templates for different conversation types.

### Export Formats
Add support for PDF, Markdown, or other export formats.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

---

## 🧠 Built by Zain Khan

From simple conversation exports to a comprehensive AI memory system.



