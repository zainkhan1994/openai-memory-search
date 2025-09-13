#!/bin/bash

# OpenAI Memory Search - Enhanced Setup Script
echo "🧠 Setting up OpenAI Memory Search - Enhanced"
echo "============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $python_version detected"

# Create flattened_output directory if it doesn't exist
if [ ! -d "flattened_output" ]; then
    mkdir -p flattened_output
    echo "📁 Created flattened_output directory"
fi

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "🔑 Creating .env template..."
    cat > .env << EOL
# OpenAI API Key (required for embeddings)
OPENAI_API_KEY=your_api_key_here

# Optional: Customize models and settings
# EMBED_MODEL=text-embedding-3-small
# CHAT_MODEL=gpt-3.5-turbo
EOL
    echo "⚠️  Please edit .env and add your OpenAI API key"
fi

# Create sample data if conversations.jsonl doesn't exist
if [ ! -f "flattened_output/conversations.jsonl" ]; then
    echo "📝 Creating sample conversation data..."
    cat > flattened_output/conversations.jsonl << EOL
{"content": "I'm thinking about building a semantic search system for my OpenAI conversations. What's the best approach?", "role": "user", "conversation_id": "conv_001", "message_id": "msg_001", "timestamp": "1703000000"}
{"content": "A great approach would be to use embeddings with FAISS for fast similarity search. You'd embed each conversation message, index them, then query with natural language.", "role": "assistant", "conversation_id": "conv_001", "message_id": "msg_002", "timestamp": "1703000060"}
{"content": "How do I implement a mind map for visualizing conversation topics?", "role": "user", "conversation_id": "conv_002", "message_id": "msg_003", "timestamp": "1703000120"}
{"content": "You can use libraries like D3.js, Cytoscape, or React Flow to create interactive mind maps. Group related conversations by topic clusters.", "role": "assistant", "conversation_id": "conv_002", "message_id": "msg_004", "timestamp": "1703000180"}
{"content": "What are the benefits of using FAISS over other vector databases?", "role": "user", "conversation_id": "conv_003", "message_id": "msg_005", "timestamp": "1703000240"}
{"content": "FAISS is optimized for similarity search, supports GPU acceleration, and works well offline. It's faster than many alternatives for read-heavy workloads.", "role": "assistant", "conversation_id": "conv_003", "message_id": "msg_006", "timestamp": "1703000300"}
EOL
    echo "✅ Sample conversation data created"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Replace flattened_output/conversations.jsonl with your actual conversation data"
echo "3. Run: python embed_and_index.py (to create embeddings)"
echo "4. Run: streamlit run enhanced_app.py (to start the web interface)"
echo ""
echo "💡 For CLI usage, run: python semantic_search.py"
echo ""