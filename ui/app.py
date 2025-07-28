# /ui/app.py
# A simple Streamlit dashboard to interact with the Nebula Recommendation API.

import streamlit as st
import requests
import pandas as pd

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/similar_items"

# --- Page Setup ---
st.set_page_config(
    page_title="Nebula Recommendation Engine",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Nebula Recommendation Engine")
st.write("A content-based recommendation system for electronics products.")

# --- UI Components ---

st.markdown("---")

# Use a session state to remember the last searched item
if 'last_item_id' not in st.session_state:
    st.session_state.last_item_id = "0132793040"

item_id_input = st.text_input(
    "Enter an Item ID (ASIN) to find similar products:",
    value=st.session_state.last_item_id,
    help="Try an ID like '0132793040' or 'B00005N576'."
)

find_button = st.button("Find Similar Items", type="primary")

# --- API Interaction and Display Logic ---

if find_button:
    if not item_id_input:
        st.warning("Please enter an Item ID.")
    else:
        st.session_state.last_item_id = item_id_input
        
        # Show a spinner while waiting for the API response
        with st.spinner(f"Searching for items similar to '{item_id_input}'..."):
            try:
                response = requests.get(f"{API_URL}/{item_id_input}")
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Found recommendations for item: **{data['source_item_id']}**")
                    
                    # Create a DataFrame for better display
                    df = pd.DataFrame(data['recommendations'])
                    df.rename(columns={'item_id': 'Recommended Item ID', 'distance': 'Similarity Score (Lower is Better)'}, inplace=True)
                    
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                elif response.status_code == 404:
                    st.error(f"Error: Item with ID '{item_id_input}' not found in the database.")
                else:
                    st.error(f"An error occurred. Status code: {response.status_code}")
                    st.json(response.text) # Show the raw error message
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Please ensure the FastAPI server is running. Error: {e}")