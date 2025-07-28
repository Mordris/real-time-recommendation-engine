# /ui/app.py
# The complete Streamlit dashboard for demonstrating Project Nebula.

import streamlit as st
import requests
import pandas as pd
import json

# --- Configuration ---
# The base URL for our FastAPI backend. All API calls will be directed here.
API_BASE_URL = "http://127.0.0.1:8000"

# --- Page Setup ---
# Configure the Streamlit page with a title, icon, and wide layout for a better look.
st.set_page_config(
    page_title="Nebula Recommendation Engine",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Nebula Recommendation Engine")
st.write("An end-to-end demonstration of a real-time recommendation system.")

# --- Session State Initialization ---
# Streamlit reruns the entire script on every interaction. `st.session_state` is a dictionary-like
# object that persists across these reruns. We use it to remember the last used user and item IDs,
# creating a more seamless user experience.
if 'user_id' not in st.session_state:
    st.session_state.user_id = "test-user-01"
if 'item_id' not in st.session_state:
    st.session_state.item_id = "0132793040" # A known valid item ID to start with.

st.markdown("---")

# --- Main Real-Time Demo ---
# Use columns to create a side-by-side layout for the interaction simulator and the results viewer.
# The ratio [1, 2] makes the right column twice as wide as the left one.
col1, col2 = st.columns([1, 2]) 

# --- Column 1: Interaction Simulator ---
with col1:
    st.subheader("Step 1: Simulate User Interaction")
    
    # Text input widgets for user and item IDs. Their values are tied to st.session_state.
    user_id_input = st.text_input("User ID", value=st.session_state.user_id, key="user_id")
    item_id_input = st.text_input("Item ID (ASIN)", value=st.session_state.item_id, key="item_id")

    # When this button is clicked, the code inside this block runs.
    if st.button("Send Interaction", type="primary"):
        if user_id_input and item_id_input:
            # Prepare the JSON payload for the POST request.
            payload = {
                "user_id": user_id_input,
                "item_id": item_id_input,
                "event_type": "click" # For this demo, we'll hardcode the event type.
            }
            try:
                # Send the interaction event to the FastAPI backend.
                response = requests.post(f"{API_BASE_URL}/interaction", json=payload)
                if response.status_code == 200:
                    st.success(f"Successfully sent interaction for user '{user_id_input}'.")
                else:
                    # Display an error message if the API returns a non-200 status.
                    st.error(f"API Error (Status {response.status_code}): {response.text}")
            except requests.exceptions.RequestException as e:
                # Display an error if the API server is not reachable.
                st.error(f"Connection Error: Could not connect to the API. Is it running? Details: {e}")
        else:
            st.warning("Please provide both a User ID and an Item ID.")

# --- Column 2: Recommendation Viewer ---
with col2:
    st.subheader("Step 2: View Real-Time Recommendations")
    # Display the user ID currently being viewed, pulled from the session state.
    st.write(f"Showing recommendations for user: **{st.session_state.user_id}**")
    
    # This button's main purpose is to trigger a rerun of the Streamlit script.
    # When clicked, the code below this point will execute again, fetching fresh data.
    if st.button("Refresh Recommendations"):
        pass

    try:
        # Fetch the latest recommendations for the current user from the API.
        response = requests.get(f"{API_BASE_URL}/recommendations/{st.session_state.user_id}")
        if response.status_code == 200:
            data = response.json()
            recs = data.get("recommendations", [])
            
            # If the user has no recommendations yet, show an informative message.
            if not recs:
                st.info("No recommendations found for this user yet. Send an interaction event to generate them.")
            else:
                # If recommendations are found, display them in a clean table.
                st.success("Current recommendations:")
                df = pd.DataFrame(recs, columns=["Recommended Item ID"])
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error(f"API Error (Status {response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: Could not connect to the API. Details: {e}")

# --- Static Similarity (Original Functionality) ---
st.markdown("---")
# `st.expander` creates a collapsible section to keep the main UI clean.
with st.expander("Bonus: Find Static Similar Items (Content-Based)"):
    static_item_id = st.text_input("Enter an Item ID (ASIN) to find similar items based on content:", value="B00005N576")
    if st.button("Find Similar"):
        if static_item_id:
            # `st.spinner` shows a loading message while the API call is in progress.
            with st.spinner(f"Searching for items similar to '{static_item_id}'..."):
                try:
                    # Call the static similarity endpoint.
                    response = requests.get(f"{API_BASE_URL}/similar_items/{static_item_id}")
                    if response.status_code == 200:
                        data = response.json()
                        df = pd.DataFrame(data['recommendations'])
                        # Rename columns for a more user-friendly display.
                        df.rename(columns={'item_id': 'Recommended Item ID', 'distance': 'Similarity Score (Lower is Better)'}, inplace=True)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.error(f"API Error (Status {response.status_code}): {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: Could not connect to the API. Details: {e}")