# /ui/app.py
# The complete Streamlit dashboard for demonstrating Project Nebula.

import streamlit as st
import requests
import pandas as pd
import json

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

# --- Page Setup ---
st.set_page_config(
    page_title="Nebula Recommendation Engine",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Nebula Recommendation Engine")
st.write("An end-to-end demonstration of a real-time recommendation system.")

# --- Session State Initialization ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = "test-user-01"
if 'item_id' not in st.session_state:
    st.session_state.item_id = "0132793040" # A known valid item ID

st.markdown("---")

# --- Main Real-Time Demo ---
col1, col2 = st.columns([1, 2]) # Create two columns

# --- Column 1: Interaction Simulator ---
with col1:
    st.subheader("Step 1: Simulate User Interaction")
    
    user_id_input = st.text_input("User ID", value=st.session_state.user_id, key="user_id")
    item_id_input = st.text_input("Item ID (ASIN)", value=st.session_state.item_id, key="item_id")

    if st.button("Send Interaction", type="primary"):
        if user_id_input and item_id_input:
            payload = {
                "user_id": user_id_input,
                "item_id": item_id_input,
                "event_type": "click" # We'll just use 'click' for the demo
            }
            try:
                response = requests.post(f"{API_BASE_URL}/interaction", json=payload)
                if response.status_code == 200:
                    st.success(f"Successfully sent interaction for user '{user_id_input}'.")
                else:
                    st.error(f"API Error (Status {response.status_code}): {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: Could not connect to the API. Is it running? Details: {e}")
        else:
            st.warning("Please provide both a User ID and an Item ID.")

# --- Column 2: Recommendation Viewer ---
with col2:
    st.subheader("Step 2: View Real-Time Recommendations")
    st.write(f"Showing recommendations for user: **{st.session_state.user_id}**")
    
    if st.button("Refresh Recommendations"):
        # This button click forces Streamlit to rerun the script, fetching new data
        pass

    try:
        response = requests.get(f"{API_BASE_URL}/recommendations/{st.session_state.user_id}")
        if response.status_code == 200:
            data = response.json()
            recs = data.get("recommendations", [])
            if not recs:
                st.info("No recommendations found for this user yet. Send an interaction event to generate them.")
            else:
                st.success("Current recommendations:")
                df = pd.DataFrame(recs, columns=["Recommended Item ID"])
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error(f"API Error (Status {response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: Could not connect to the API. Details: {e}")

# --- Static Similarity (Original Functionality) ---
st.markdown("---")
with st.expander("Bonus: Find Static Similar Items (Content-Based)"):
    static_item_id = st.text_input("Enter an Item ID (ASIN) to find similar items based on content:", value="B00005N576")
    if st.button("Find Similar"):
        if static_item_id:
            with st.spinner(f"Searching for items similar to '{static_item_id}'..."):
                try:
                    response = requests.get(f"{API_BASE_URL}/similar_items/{static_item_id}")
                    if response.status_code == 200:
                        data = response.json()
                        df = pd.DataFrame(data['recommendations'])
                        df.rename(columns={'item_id': 'Recommended Item ID', 'distance': 'Similarity Score (Lower is Better)'}, inplace=True)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.error(f"API Error (Status {response.status_code}): {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: Could not connect to the API. Details: {e}")