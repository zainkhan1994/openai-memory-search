import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import time

# --- Configuration ---
load_dotenv() # Loads environment variables from .env file

# --- API Key Check ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file. Please create .env and set it.")
    exit()

client = OpenAI(api_key=api_key)

# --- File Paths ---
BASE_OUTPUT_DIR = "flattened_output" # Ensure this directory exists
METADATA_PATH = os.path.join(BASE_OUTPUT_DIR, "zain_metadata.json") # Corrected from your previous streamlit_app.py which used zain_metadata.json
OUTPUT_INSIGHTS_PATH = os.path.join(BASE_OUTPUT_DIR, "precalculated_insights.json")

# --- Model Configuration ---
CHAT_MODEL_FOR_SUMMARY = "gpt-3.5-turbo" # Default: cost-effective and fast
# CHAT_MODEL_FOR_SUMMARY = "gpt-4-turbo-preview" # Alternative: higher quality, higher cost, slower

# --- Test Configuration ---
# Set TEST_SUBSET_SIZE to a small number (e.g., 3, 5, 10) for initial testing.
# Set to None or a very large number (or comment out the test block) for a full run.
TEST_SUBSET_SIZE = 5 # <<< CHANGE THIS FOR MORE/LESS TESTING, OR COMMENT OUT BLOCK BELOW

# --- Helper Functions ---
def get_full_conversation_text_for_llm(conversation_id, all_metadata_list):
    """
    Extracts and formats the text of a full conversation for LLM processing.
    """
    # Filter messages for the given conversation_id and ensure they have a timestamp for sorting
    conversation_messages = [
        msg for msg in all_metadata_list
        if msg.get("conversation_id") == conversation_id and msg.get("timestamp") is not None
    ]
    
    if not conversation_messages:
        # print(f"  Debug: No messages found or no timestamps for conv_id: {conversation_id}")
        return None

    # Sort messages by timestamp (converted to float for safety)
    try:
        conversation_messages.sort(key=lambda x: float(x["timestamp"]))
    except (TypeError, ValueError) as e:
        print(f"  Warning: Could not sort messages for conv_id {conversation_id} due to timestamp issue: {e}. Skipping this conversation for LLM processing.")
        return None
    
    llm_input_parts = []
    for msg in conversation_messages:
        role_prefix = "User" if msg.get('role') == 'user' else "Assistant"
        content = msg.get('content', '') # Ensure content is a string, default to empty if None
        if not isinstance(content, str): # If content is not a string (e.g. dict from bad data)
            content = str(content) # Attempt to convert to string
        llm_input_parts.append(f"{role_prefix}: {content.strip()}") # Strip whitespace
        
    return "\n".join(llm_input_parts)

def generate_insights_for_text(conversation_text, conversation_id_for_debug="N/A"):
    """
    Sends conversation text to OpenAI to get a one-sentence summary and 5 keywords.
    """
    if not conversation_text or not conversation_text.strip():
        print(f"  Debug (conv_id: {conversation_id_for_debug}): Empty conversation text provided to generate_insights_for_text. Skipping LLM call.")
        return None, None
        
    try:
        system_prompt = (
            "You are an expert at analyzing conversation transcripts. "
            "Your task is to provide a concise one-sentence summary of the entire conversation "
            "and then list exactly 5 distinct and most important keywords or keyphrases from it. "
            "Format your response strictly as follows, with each part on a new line:\n"
            "SUMMARY: [Your one-sentence summary here]\n"
            "KEYWORDS: [keyword1, keyword2, keyword3, keyword4, keyword5]"
        )
        
        # print(f"  Debug (conv_id: {conversation_id_for_debug}): Sending text to LLM starting with: {conversation_text[:200]}...")

        response = client.chat.completions.create(
            model=CHAT_MODEL_FOR_SUMMARY,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.2, # Lower temperature for more factual/deterministic output
            max_tokens=250 # Increased slightly to ensure full summary and keywords fit
        )
        content = response.choices[0].message.content
        
        # print(f"  Debug (conv_id: {conversation_id_for_debug}): LLM Raw Response:\n{content}")

        summary_line = ""
        keywords_line = ""

        for line in content.split('\n'):
            if line.upper().startswith("SUMMARY:"):
                summary_line = line.replace("SUMMARY:", "").strip()
            elif line.upper().startswith("KEYWORDS:"):
                keywords_line = line.replace("KEYWORDS:", "").strip()
        
        keywords_list = [kw.strip() for kw in keywords_line.split(',') if kw.strip()]
        
        if not summary_line and not keywords_list:
            print(f"  Warning (conv_id: {conversation_id_for_debug}): LLM returned an empty or non-parsable response for summary/keywords. Raw: {content}")
            return None, None # Indicate failure to parse

        return summary_line, keywords_list

    except Exception as e:
        print(f"  ERROR (conv_id: {conversation_id_for_debug}): During OpenAI API call or parsing: {e}")
        return None, None

def main():
    print(f"--- Starting Batch Insight Generation ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"Using LLM model: {CHAT_MODEL_FOR_SUMMARY}")

    # Ensure base output directory exists
    if not os.path.exists(BASE_OUTPUT_DIR):
        print(f"ERROR: Base output directory '{BASE_OUTPUT_DIR}' not found. Please ensure your data is structured correctly.")
        return

    # 1. Load all metadata
    try:
        with open(METADATA_PATH, "r") as f:
            all_metadata_list = json.load(f)
        if not all_metadata_list:
            print(f"ERROR: Metadata file at {METADATA_PATH} is empty or could not be loaded correctly.")
            return
        print(f"Successfully loaded {len(all_metadata_list)} total messages from {METADATA_PATH}.")
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {METADATA_PATH}. Please run `embed_and_index.py` first.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {METADATA_PATH}. The file might be corrupt.")
        return

    # 2. Load existing insights (to avoid reprocessing and allow resuming)
    existing_insights = {}
    if os.path.exists(OUTPUT_INSIGHTS_PATH):
        try:
            with open(OUTPUT_INSIGHTS_PATH, "r") as f:
                existing_insights = json.load(f)
            print(f"Loaded {len(existing_insights)} existing pre-calculated insights from {OUTPUT_INSIGHTS_PATH}.")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing insights file at {OUTPUT_INSIGHTS_PATH}. It might be recreated if new insights are generated.")
            existing_insights = {} # Start fresh if corrupt
    
    processed_conv_ids_in_this_run = set(existing_insights.keys())
    
    # 3. Get unique conversation IDs from the metadata
    # Ensure conversation_id is not None before adding to the set
    unique_conversation_ids_all = sorted(list(set(
        msg.get("conversation_id") for msg in all_metadata_list if msg.get("conversation_id") is not None
    )))
    
    if not unique_conversation_ids_all:
        print("ERROR: No valid 'conversation_id' found in any metadata records. Cannot proceed.")
        return
    print(f"Found {len(unique_conversation_ids_all)} unique conversation IDs in metadata.")

    # --- FOR TESTING: Limit to a small number of conversations ---
    ids_to_process = unique_conversation_ids_all
    if TEST_SUBSET_SIZE is not None and len(unique_conversation_ids_all) > TEST_SUBSET_SIZE:
        print(f"---!!! ATTENTION: RUNNING IN TEST MODE - TARGETING {TEST_SUBSET_SIZE} CONVERSATIONS (if not already processed) !!!---")
        # Select a subset that hasn't been processed yet, if possible, or just the first few
        unprocessed_ids = [cid for cid in unique_conversation_ids_all if cid not in processed_conv_ids_in_this_run]
        if len(unprocessed_ids) >= TEST_SUBSET_SIZE:
            ids_to_process = unprocessed_ids[:TEST_SUBSET_SIZE]
        elif unprocessed_ids: # if there are some unprocessed, but fewer than TEST_SUBSET_SIZE
            ids_to_process = unprocessed_ids
            print(f"--- NOTE: Fewer than {TEST_SUBSET_SIZE} unprocessed conversations remaining. Processing {len(ids_to_process)}.")
        else: # if all are processed or no unprocessed ones to meet the test size
            ids_to_process = unique_conversation_ids_all[:TEST_SUBSET_SIZE] # Fallback to first few for re-test if needed
            if processed_conv_ids_in_this_run.issuperset(set(ids_to_process)):
                 print(f"--- NOTE: The first {TEST_SUBSET_SIZE} conversations appear to be already processed. To re-process, clear '{OUTPUT_INSIGHTS_PATH}'. ---")

    elif TEST_SUBSET_SIZE is not None:
        print(f"--- NOTE: TEST_SUBSET_SIZE ({TEST_SUBSET_SIZE}) is set, but total unique conversations ({len(unique_conversation_ids_all)}) is less than or equal. Processing all available. ---")
    # --- END OF TEST MODIFICATION BLOCK ---

    if not ids_to_process:
        print("No conversation IDs selected for processing (possibly all processed in test mode). Exiting.")
        return

    print(f"Will attempt to process insights for {len(ids_to_process)} conversation IDs.")
    new_insights_generated_this_session = 0
    
    for i, conv_id in enumerate(ids_to_process):
        if conv_id in existing_insights: # Check against insights loaded at start
            print(f"({i+1}/{len(ids_to_process)}) Skipping already processed conversation ID (found in loaded insights): {conv_id}")
            continue

        print(f"({i+1}/{len(ids_to_process)}) Processing conversation ID: {conv_id}")
        
        # Get the full text for this conversation
        conversation_text = get_full_conversation_text_for_llm(conv_id, all_metadata_list)

        if not conversation_text:
            print(f"  No processable text found for conversation ID: {conv_id}. Marking as processed and skipping.")
            existing_insights[conv_id] = {"summary": "Error: No processable text", "keywords": []} # Log error
            continue # Move to next conversation ID

        # Generate summary and keywords
        summary, keywords = generate_insights_for_text(conversation_text, conv_id)

        if summary is not None or keywords is not None: # Even if one is None but the other exists
            existing_insights[conv_id] = {
                "summary": summary if summary is not None else "Generation failed or N/A", 
                "keywords": keywords if keywords is not None else []
            }
            new_insights_generated_this_session += 1
            print(f"  Generated insights for {conv_id}.")
            if summary: print(f"    Summary: '{summary[:70].replace(chr(10), ' ')}...'") # Show a bit of the summary
            if keywords: print(f"    Keywords: {keywords}")
        else:
            print(f"  Failed to generate valid insights for {conv_id}. It will be marked to avoid retries in this run.")
            # Optionally, mark it with an error state in existing_insights if you want to track failures persistently
            existing_insights[conv_id] = {"summary": "Error: Failed to generate", "keywords": []}


        # Save incrementally to avoid losing all progress on error or interruption
        # Save every 5 processed conversations in this session or at the very end
        if new_insights_generated_this_session > 0 and (new_insights_generated_this_session % 5 == 0 or i == len(ids_to_process) - 1):
            try:
                with open(OUTPUT_INSIGHTS_PATH, "w") as f:
                    json.dump(existing_insights, f, indent=2)
                print(f"  -- Successfully saved {len(existing_insights)} total insights to {OUTPUT_INSIGHTS_PATH} --")
            except Exception as e:
                print(f"  ERROR saving insights incrementally: {e}")
        
        # OpenAI API rate limiting: be respectful. Adjust sleep time as needed.
        # For gpt-3.5-turbo, 1 second is usually fine. For gpt-4, you might need more.
        time.sleep(1.1) # Slightly more than 1s to be safe with potential bursts

    print(f"\n--- Batch Processing Session Finished ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"Generated new insights for {new_insights_generated_this_session} conversations in this session.")
    print(f"Total insights now stored in {OUTPUT_INSIGHTS_PATH}: {len(existing_insights)}")

if __name__ == "__main__":
    # Ensure the flattened_output directory exists before trying to write to it
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        print(f"Created directory: {BASE_OUTPUT_DIR}")
    main()
