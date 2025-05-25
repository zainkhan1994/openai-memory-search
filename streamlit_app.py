import streamlit as st
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
import re # For creating a safe filename

# --- Configuration and Setup ---
st.set_page_config(page_title="Zain's Mind Search", layout="wide")
st.title("🧠 Zain's Mind Search")
st.markdown("Ask questions about your past conversations and discover what you've thought, said, or learned.")

# Load environment variables (for OpenAI API key)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in .env file. Please ensure it's set.")
    st.stop()

client = OpenAI(api_key=api_key)

# Paths to your data
METADATA_PATH = "flattened_output/zain_metadata.json"
INDEX_PATH = "flattened_output/zain_index.faiss"
# Path for pre-calculated insights (will be used by batch script later)
PRECALCULATED_INSIGHTS_PATH = "flattened_output/precalculated_insights.json" 

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL_FOR_SUMMARY = "gpt-3.5-turbo"

# --- Caching Functions to Load Data Once ---
@st.cache_resource
def load_all_metadata_list_version():
    try:
        with open(METADATA_PATH, "r") as f:
            metadata_list = json.load(f)
        for record in metadata_list:
            if record.get("timestamp"):
                try:
                    record["datetime_obj"] = datetime.fromtimestamp(float(record["timestamp"]))
                except (ValueError, TypeError): record["datetime_obj"] = None
            else: record["datetime_obj"] = None
        return metadata_list
    except FileNotFoundError: st.error(f"Metadata file not found: {METADATA_PATH}. Did you run embed_and_index.py?"); return []
    except json.JSONDecodeError: st.error(f"Error decoding {METADATA_PATH}. File might be corrupt."); return []

@st.cache_resource
def load_faiss_index():
    try:
        return faiss.read_index(INDEX_PATH)
    except Exception as e:
        st.error(f"FAISS index file not found or could not be read: {INDEX_PATH}. Error: {e}"); return None

@st.cache_resource
def load_precalculated_insights():
    """Loads pre-calculated summaries and keywords if the file exists."""
    if os.path.exists(PRECALCULATED_INSIGHTS_PATH):
        try:
            with open(PRECALCULATED_INSIGHTS_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Could not decode {PRECALCULATED_INSIGHTS_PATH}. On-demand generation will be used.")
            return {} # Return empty dict if file is corrupt
        except Exception as e:
            st.warning(f"Error loading {PRECALCULATED_INSIGHTS_PATH}: {e}. On-demand generation will be used.")
            return {}
    return {} # Return empty dict if file doesn't exist

# --- Helper Functions ---
def get_embedding(text, model=EMBED_MODEL):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding).astype("float32")
    except Exception as e:
        st.error(f"Error getting embedding: {e}"); return None

@st.cache_data # Cache based on arguments
def get_full_conversation_messages(_conversation_id, _all_metadata_list):
    """Retrieves all messages for a given conversation_id, sorted by timestamp."""
    if not _all_metadata_list or not _conversation_id: return []
    
    conversation_messages = [
        msg for msg in _all_metadata_list
        if msg.get("conversation_id") == _conversation_id and msg.get("datetime_obj") is not None
    ]
    conversation_messages.sort(key=lambda x: x["datetime_obj"])
    return conversation_messages

@st.cache_data
def generate_summary_and_keywords_on_demand(_conversation_text_for_llm, _conversation_id):
    """On-demand summary and keyword generation if not pre-calculated."""
    if not _conversation_text_for_llm: return None, None
    try:
        system_prompt = (
            "You are an expert at summarizing conversations and extracting key topics. "
            "Analyze the following conversation transcript. "
            "Provide a concise one-sentence summary of the entire conversation. "
            "Then, list the 5 most important and distinct keywords or keyphrases from the conversation. "
            "Format your response as follows:\n"
            "SUMMARY: [Your one-sentence summary here]\n"
            "KEYWORDS: [keyword1, keyword2, keyword3, keyword4, keyword5]"
        )
        response = client.chat.completions.create(
            model=CHAT_MODEL_FOR_SUMMARY,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _conversation_text_for_llm}
            ],
            temperature=0.3, max_tokens=200
        )
        content = response.choices[0].message.content
        summary_line, keywords_line = "", ""
        for line in content.split('\n'):
            if line.upper().startswith("SUMMARY:"): summary_line = line.replace("SUMMARY:", "").strip()
            elif line.upper().startswith("KEYWORDS:"): keywords_line = line.replace("KEYWORDS:", "").strip()
        keywords_list = [kw.strip() for kw in keywords_line.split(',') if kw.strip()] if keywords_line else []
        return summary_line, keywords_list
    except Exception as e:
        st.error(f"Error generating on-demand insights for thread {_conversation_id}: {e}"); return None, None

def format_timestamp(dt_obj):
    if dt_obj: return dt_obj.strftime('%a - %b %d @ %I:%M %p').upper()
    return "No Timestamp"

def format_role_for_display(role, dt_obj):
    formatted_ts = format_timestamp(dt_obj)
    if role == 'user': return f"Zain asked on {formatted_ts}"
    return f"OpenAI Model responded on {formatted_ts}"

def create_safe_filename(base_name):
    """Creates a safe filename by removing invalid characters."""
    base_name = re.sub(r'[^\w\s-]', '', base_name) # Remove non-alphanumeric, non-space, non-hyphen
    base_name = re.sub(r'[-\s]+', '-', base_name).strip('-_') # Replace spaces/hyphens with single hyphen
    return f"{base_name[:50]}.txt" # Truncate and add extension

# --- Load Data ---
metadata_list_for_faiss = load_all_metadata_list_version()
index = load_faiss_index()
precalculated_insights = load_precalculated_insights() # Load pre-calculated insights

if not metadata_list_for_faiss or not index:
    st.warning("Core data (metadata or FAISS index) not loaded. Please check errors and ensure `embed_and_index.py` ran successfully.")
    st.stop()

# --- Streamlit App UI ---
# Sidebar for filters
st.sidebar.header("🔎 Filter Options")
filter_role = st.sidebar.selectbox(
    "Filter by Role (in hit message):",
    ["Any", "User (Zain)", "Assistant (OpenAI)"],
    key="filter_role_select"
)

search_query = st.text_input("🔍 Ask your brain:", placeholder="e.g., when did I talk about Apollo and EPM?")

if search_query:
    st.markdown("---")
    st.subheader("⏳ Searching your memory...")

    query_vector = get_embedding(search_query)

    if query_vector is not None:
        try:
            D, I = index.search(np.array([query_vector]), k=20) # Fetch more results initially for filtering
            
            if I[0].size == 0 or I[0][0] == -1:
                 st.info("No relevant thoughts initially found for your query. Try rephrasing.")
            else:
                results_to_display = []
                for i, idx in enumerate(I[0]):
                    if idx < 0 or idx >= len(metadata_list_for_faiss): continue
                    
                    hit_message = metadata_list_for_faiss[idx]
                    score = D[0][i]

                    # Apply role filter
                    if filter_role == "User (Zain)" and hit_message.get('role') != 'user':
                        continue
                    if filter_role == "Assistant (OpenAI)" and hit_message.get('role') != 'assistant':
                        continue
                    
                    results_to_display.append({"hit_message": hit_message, "score": score, "original_faiss_idx": idx})

                if not results_to_display:
                    st.info(f"No results match your query combined with the role filter: '{filter_role}'.")
                else:
                    st.subheader(f"💡 Top Matching Thoughts (Filtered by Role: {filter_role})")
                    for i, res_data in enumerate(results_to_display[:10]): # Display top 10 after filtering
                        hit_message = res_data["hit_message"]
                        score = res_data["score"]
                        original_faiss_idx = res_data["original_faiss_idx"] # For unique keys

                        st.markdown(f"**Result {i+1} (Score: {score:.2f})**")
                        
                        hit_display_role_ts = format_role_for_display(hit_message.get('role'), hit_message.get('datetime_obj'))
                        st.markdown(f"🎯 **Hit Message ({hit_display_role_ts}):** {hit_message.get('content')}")

                        conversation_id = hit_message.get("conversation_id")
                        if conversation_id:
                            expander_title = f"🔍 View Full Conversation & Insights (ID: {conversation_id}, Hit: '{hit_message.get('content', '')[:30].strip()}...')"
                            with st.expander(expander_title):
                                full_thread_messages = get_full_conversation_messages(conversation_id, metadata_list_for_faiss)
                                conversation_text_for_llm_parts = []

                                if full_thread_messages:
                                    st.markdown("##### Entire Conversation Thread:")
                                    for ctx_msg in full_thread_messages:
                                        display_role_ts_ctx = format_role_for_display(ctx_msg.get('role'), ctx_msg.get('datetime_obj'))
                                        
                                        if ctx_msg.get("message_id") == hit_message.get("message_id"):
                                            st.markdown(f"**➡️ {display_role_ts_ctx}: {ctx_msg.get('content')}**")
                                        else:
                                            st.markdown(f"    {display_role_ts_ctx}: {ctx_msg.get('content')}")
                                        
                                        simple_role_prefix = "User" if ctx_msg.get('role') == 'user' else "Assistant"
                                        conversation_text_for_llm_parts.append(f"{simple_role_prefix}: {ctx_msg.get('content')}")
                                    
                                    st.markdown("---")
                                    # Insights Section
                                    st.markdown("##### ✨ Insights for this Thread:")
                                    summary, keywords = None, None
                                    # Check for pre-calculated insights first
                                    if conversation_id in precalculated_insights:
                                        insights = precalculated_insights[conversation_id]
                                        summary = insights.get("summary")
                                        keywords = insights.get("keywords")
                                        st.caption("(Insights loaded from pre-calculated data)")
                                    
                                    if not summary and not keywords: # If not pre-calculated or failed to load
                                        if st.button(f"Generate Insights Now for Thread {conversation_id}", key=f"ondemand_insight_{conversation_id}_{original_faiss_idx}"):
                                            with st.spinner("Generating insights on-demand..."):
                                                full_conversation_text = "\n".join(conversation_text_for_llm_parts)
                                                summary, keywords = generate_summary_and_keywords_on_demand(full_conversation_text, conversation_id)
                                    
                                    if summary: st.markdown(f"**📝 Summary:** {summary}")
                                    else: st.write("No summary available or generated yet for this thread.")
                                    
                                    if keywords: st.markdown(f"**🔑 Keywords:** {', '.join(keywords)}")
                                    else: st.write("No keywords available or generated yet for this thread.")

                                    # Export Button
                                    st.markdown("---")
                                    full_conversation_text_for_export = "\n\n".join(
                                        [f"{format_role_for_display(msg.get('role'), msg.get('datetime_obj'))}:\n{msg.get('content')}" for msg in full_thread_messages]
                                    )
                                    if summary: full_conversation_text_for_export += f"\n\n--- SUMMARY ---\n{summary}"
                                    if keywords: full_conversation_text_for_export += f"\n\n--- KEYWORDS ---\n{', '.join(keywords)}"
                                    
                                    export_filename_base = f"conversation_{conversation_id}"
                                    if full_thread_messages:
                                        # Use first few words of first user message for a more descriptive filename
                                        first_user_msg_content = next((msg.get("content") for msg in full_thread_messages if msg.get("role") == "user"), None)
                                        if first_user_msg_content:
                                            export_filename_base = f"conv_{conversation_id}_{first_user_msg_content[:20]}"

                                    st.download_button(
                                        label="📥 Export this Thread (Text)",
                                        data=full_conversation_text_for_export,
                                        file_name=create_safe_filename(export_filename_base),
                                        mime="text/plain",
                                        key=f"export_{conversation_id}_{original_faiss_idx}"
                                    )
                                else:
                                    st.write("Could not retrieve full conversation for this message.")
                        st.markdown("---")
        except Exception as e:
            st.error(f"An error occurred during search: {e}")
    else:
        st.error("Could not process your query embedding. Please try again.")
else:
    st.info("Type a question above to start searching your thoughts, or use filters in the sidebar.")
