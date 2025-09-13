#!/usr/bin/env python3
"""
Demo script showcasing the enhanced OpenAI Memory Search features
"""

import json
import os
from datetime import datetime

def main():
    print("🧠 OpenAI Memory Search - Enhanced Demo")
    print("=" * 50)
    
    # Check if setup is complete
    required_files = [
        "flattened_output/conversations.jsonl",
        "flattened_output/zain_metadata.json",
        ".env"
    ]
    
    print("\n📋 Checking setup...")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
    
    # Load and display sample data
    if os.path.exists("flattened_output/zain_metadata.json"):
        try:
            with open("flattened_output/zain_metadata.json", "r") as f:
                metadata = json.load(f)
            
            print(f"\n📊 Data Statistics:")
            print(f"Total messages: {len(metadata)}")
            
            conversations = set(m.get('conversation_id') for m in metadata)
            print(f"Total conversations: {len(conversations)}")
            
            user_msgs = len([m for m in metadata if m.get('role') == 'user'])
            assistant_msgs = len([m for m in metadata if m.get('role') == 'assistant'])
            print(f"User messages: {user_msgs}")
            print(f"Assistant messages: {assistant_msgs}")
            
            # Show sample topics
            topics = {}
            for msg in metadata:
                topic = msg.get('topic', 'general')
                topics[topic] = topics.get(topic, 0) + 1
            
            print(f"\n🏷️ Topics found:")
            for topic, count in topics.items():
                print(f"  {topic}: {count} messages")
                
        except Exception as e:
            print(f"❌ Error reading metadata: {e}")
    
    print(f"\n🚀 Features Available:")
    print("1. 🔍 Semantic Search - Natural language queries")
    print("2. 🗺️ Mind Maps - Visual topic exploration")
    print("3. 📝 Notes - Add personal annotations")
    print("4. 📊 Analytics - Conversation insights")
    
    print(f"\n🖥️ To launch the web interface:")
    print("streamlit run enhanced_app.py")
    
    print(f"\n🔧 To use CLI tools:")
    print("python semantic_search.py")
    
    print(f"\n💡 Quick Demo Commands:")
    print("./setup.sh                    # Complete setup")
    print("python embed_and_index.py     # Create embeddings")
    print("streamlit run enhanced_app.py # Launch web UI")

if __name__ == "__main__":
    main()