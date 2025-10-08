import streamlit as st
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
from collections import defaultdict
import uuid

# --- Configuration and Setup ---
st.set_page_config(
    page_title="OpenAI Memory Search - Enhanced", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .search-results {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .conversation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .note-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>🧠 OpenAI Memory Search</h1>
    <p>Enhanced Semantic Search with Mind Maps & Notes</p>
</div>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Constants
METADATA_PATH = "flattened_output/zain_metadata.json"
INDEX_PATH = "flattened_output/zain_index.faiss"
NOTES_PATH = "flattened_output/conversation_notes.json"
EMBED_MODEL = "text-embedding-3-small"

# --- Helper Functions ---
@st.cache_resource
def load_metadata():
    try:
        with open(METADATA_PATH, "r") as f:
            metadata_list = json.load(f)
        for record in metadata_list:
            if record.get("timestamp"):
                try:
                    record["datetime_obj"] = datetime.fromtimestamp(float(record["timestamp"]))
                except (ValueError, TypeError):
                    record["datetime_obj"] = None
        return metadata_list
    except FileNotFoundError:
        st.error(f"Metadata file not found: {METADATA_PATH}")
        return []

@st.cache_resource
def load_faiss_index():
    try:
        return faiss.read_index(INDEX_PATH)
    except Exception as e:
        st.error(f"FAISS index not found: {INDEX_PATH}")
        return None

def load_notes():
    try:
        with open(NOTES_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_notes(notes):
    with open(NOTES_PATH, "w") as f:
        json.dump(notes, f, indent=2)

def get_embedding(text, model=EMBED_MODEL):
    if not api_key or api_key == "your_api_key_here":
        st.warning("Please set a valid OpenAI API key in .env file")
        return None
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding).astype("float32")
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None

def cluster_conversations(metadata_list, n_clusters=5):
    """Create topic clusters for mind map visualization"""
    if not metadata_list:
        return [], []
    
    # Create simple topic clusters based on keywords
    topics = defaultdict(list)
    keywords_map = {
        'search': ['search', 'query', 'find', 'semantic'],
        'ai': ['ai', 'openai', 'model', 'assistant'],
        'database': ['database', 'faiss', 'vector', 'index'],
        'ui': ['ui', 'interface', 'user', 'design'],
        'notes': ['note', 'annotation', 'comment', 'tag']
    }
    
    for i, record in enumerate(metadata_list):
        content = record.get('content', '').lower()
        assigned = False
        for topic, keywords in keywords_map.items():
            if any(keyword in content for keyword in keywords):
                topics[topic].append(i)
                assigned = True
                break
        if not assigned:
            topics['general'].append(i)
    
    return topics, keywords_map

def create_mind_map(topics, metadata_list):
    """Create a network graph for mind map visualization"""
    fig = go.Figure()
    
    # Create network graph
    G = nx.Graph()
    
    # Add central node
    G.add_node("Memory Search", type="central")
    
    # Add topic nodes and conversation nodes
    colors = px.colors.qualitative.Set3
    node_trace = []
    edge_trace = []
    
    pos_data = {}
    pos_data["Memory Search"] = (0, 0)
    
    # Add topic nodes around center
    angle_step = 2 * np.pi / len(topics)
    for i, (topic, conv_indices) in enumerate(topics.items()):
        angle = i * angle_step
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        pos_data[topic] = (x, y)
        G.add_node(topic, type="topic")
        G.add_edge("Memory Search", topic)
        
        # Add conversation nodes around each topic
        conv_angle_step = 2 * np.pi / max(len(conv_indices), 1)
        for j, conv_idx in enumerate(conv_indices[:5]):  # Limit to 5 conversations per topic
            if conv_idx < len(metadata_list):
                conv_angle = j * conv_angle_step
                conv_x = x + 1.5 * np.cos(conv_angle)
                conv_y = y + 1.5 * np.sin(conv_angle)
                conv_title = metadata_list[conv_idx].get('content', '')[:30] + "..."
                node_id = f"conv_{conv_idx}"
                pos_data[node_id] = (conv_x, conv_y)
                G.add_node(node_id, type="conversation", title=conv_title)
                G.add_edge(topic, node_id)
    
    # Create traces for edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos_data[edge[0]]
        x1, y1 = pos_data[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Create node traces for different types
    for node_type in ["central", "topic", "conversation"]:
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            if G.nodes[node].get('type') == node_type:
                x, y = pos_data[node]
                node_x.append(x)
                node_y.append(y)
                
                if node_type == "central":
                    node_text.append("🧠 Memory Search")
                    node_info.append("Central hub for all conversations")
                elif node_type == "topic":
                    node_text.append(f"📂 {node.title()}")
                    node_info.append(f"Topic: {node}")
                else:
                    title = G.nodes[node].get('title', node)
                    node_text.append(f"💬 {title[:20]}...")
                    node_info.append(title)
        
        size = 30 if node_type == "central" else 20 if node_type == "topic" else 15
        color = colors[0] if node_type == "central" else colors[1] if node_type == "topic" else colors[2]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            hoverinfo='text',
            marker=dict(size=size, color=color, line=dict(width=2, color='white')),
            name=node_type.title()
        ))
    
    fig.update_layout(
        title="Conversation Mind Map",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Interactive mind map of your conversations",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

# --- Load Data ---
metadata_list = load_metadata()
index = load_faiss_index()
notes = load_notes()

# --- Sidebar ---
st.sidebar.header("🔧 Tools & Filters")

# Mode selection
mode = st.sidebar.selectbox(
    "Choose Mode:",
    ["🔍 Search", "🗺️ Mind Map", "📝 Notes Manager", "📊 Analytics"]
)

if mode == "🔍 Search":
    st.header("🔍 Semantic Search")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Ask your memory:",
            placeholder="e.g., tell me about semantic search implementations"
        )
    
    with col2:
        search_button = st.button("Search", type="primary")
    
    # Filters
    with st.sidebar:
        st.subheader("🔧 Search Filters")
        filter_role = st.selectbox("Role:", ["Any", "User", "Assistant"])
        date_filter = st.date_input("Filter by date (optional):")
        max_results = st.slider("Max results:", 1, 20, 10)
    
    if search_query and (search_button or search_query):
        if metadata_list and index:
            st.markdown('<div class="search-results">', unsafe_allow_html=True)
            st.write(f"🔍 Searching for: **{search_query}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Perform search (simplified for demo)
            st.info("Search functionality requires valid OpenAI API key and embedded data.")
            
            # Show sample results for demo
            st.subheader("📋 Search Results")
            for i, record in enumerate(metadata_list[:max_results]):
                with st.container():
                    st.markdown('<div class="conversation-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([6, 2, 2])
                    
                    with col1:
                        role_icon = "👤" if record.get('role') == 'user' else "🤖"
                        st.write(f"{role_icon} **{record.get('role', 'unknown').title()}**")
                        st.write(record.get('content', '')[:200] + "...")
                    
                    with col2:
                        if record.get('datetime_obj'):
                            st.write("📅", record['datetime_obj'].strftime('%Y-%m-%d'))
                            st.write("🕐", record['datetime_obj'].strftime('%H:%M'))
                    
                    with col3:
                        conv_id = record.get('conversation_id', f'conv_{i}')
                        if st.button(f"💬 View", key=f"view_{i}"):
                            st.session_state[f'show_conv_{conv_id}'] = True
                        
                        if st.button(f"📝 Note", key=f"note_{i}"):
                            st.session_state[f'show_note_{conv_id}'] = True
                    
                    # Show full conversation if requested
                    if st.session_state.get(f'show_conv_{conv_id}', False):
                        st.write("**Full Conversation:**")
                        # Show related messages from same conversation
                        conv_messages = [r for r in metadata_list if r.get('conversation_id') == conv_id]
                        for msg in conv_messages:
                            role_icon = "👤" if msg.get('role') == 'user' else "🤖"
                            st.write(f"{role_icon} {msg.get('content', '')}")
                    
                    # Note input if requested
                    if st.session_state.get(f'show_note_{conv_id}', False):
                        note_text = st.text_area(f"Add note for conversation {conv_id}:", 
                                               value=notes.get(conv_id, ''), 
                                               key=f"note_input_{conv_id}")
                        if st.button(f"Save Note", key=f"save_note_{i}"):
                            notes[conv_id] = note_text
                            save_notes(notes)
                            st.success("Note saved!")
                            st.session_state[f'show_note_{conv_id}'] = False
                            st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No data loaded. Please run the embedding script first.")

elif mode == "🗺️ Mind Map":
    st.header("🗺️ Conversation Mind Map")
    
    if metadata_list:
        topics, keywords_map = cluster_conversations(metadata_list)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            mind_map_fig = create_mind_map(topics, metadata_list)
            st.plotly_chart(mind_map_fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Topic Summary")
            for topic, indices in topics.items():
                st.markdown(f"**{topic.title()}** ({len(indices)} conversations)")
                keywords = keywords_map.get(topic, [])
                if keywords:
                    st.write(f"Keywords: {', '.join(keywords)}")
                st.write("---")
    else:
        st.info("No conversation data available for mind map.")

elif mode == "📝 Notes Manager":
    st.header("📝 Notes Manager")
    
    if notes:
        st.subheader("Your Conversation Notes")
        for conv_id, note in notes.items():
            st.markdown('<div class="note-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"**Conversation:** {conv_id}")
                st.write(note)
            
            with col2:
                if st.button(f"🗑️ Delete", key=f"del_{conv_id}"):
                    del notes[conv_id]
                    save_notes(notes)
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No notes saved yet. Add notes from the search results!")

elif mode == "📊 Analytics":
    st.header("📊 Conversation Analytics")
    
    if metadata_list:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Conversations", len(set(r.get('conversation_id') for r in metadata_list)))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Messages", len(metadata_list))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            user_msgs = len([r for r in metadata_list if r.get('role') == 'user'])
            st.metric("User Messages", user_msgs)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Notes Saved", len(notes))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Timeline chart
        st.subheader("📈 Message Timeline")
        dates = []
        for record in metadata_list:
            if record.get('datetime_obj'):
                dates.append(record['datetime_obj'].date())
        
        if dates:
            df = pd.DataFrame({'date': dates})
            daily_counts = df.groupby('date').size().reset_index(name='count')
            fig = px.line(daily_counts, x='date', y='count', title='Messages per Day')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No date information available for timeline.")
        
        # Role distribution
        st.subheader("👥 Message Distribution")
        roles = [r.get('role', 'unknown') for r in metadata_list]
        role_counts = pd.Series(roles).value_counts()
        fig = px.pie(values=role_counts.values, names=role_counts.index, title='Messages by Role')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for analytics.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    Enhanced OpenAI Memory Search | Built with Streamlit & Love 💙
</div>
""", unsafe_allow_html=True)