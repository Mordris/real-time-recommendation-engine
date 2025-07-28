# /api/main.py
# Main FastAPI application for the Nebula Recommendation Engine.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from pymilvus import connections, utility, Collection
import sys

# Add the project root to the Python path to allow importing 'config'
sys.path.append('..')
import config

# --- Pydantic Models for API Response ---

class SimilarItem(BaseModel):
    """Defines the structure for a single similar item in the response."""
    item_id: str = Field(..., description="The unique ID of the similar item (ASIN).")
    distance: float = Field(..., description="The similarity score (L2 distance). Lower is more similar.")

class RecommendationResponse(BaseModel):
    """Defines the structure for the main API response."""
    source_item_id: str = Field(..., description="The item ID provided in the request.")
    recommendations: List[SimilarItem] = Field(..., description="A list of recommended items.")

# --- FastAPI Application Setup ---

app = FastAPI(
    title="Nebula Recommendation Engine API",
    description="Provides content-based item similarity recommendations.",
    version="0.1.0"
)

# --- Milvus Connection Management ---

@app.on_event("startup")
async def startup_event():
    """
    Connects to Milvus and loads the collection into memory on API startup.
    """
    try:
        print("Connecting to Milvus...")
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        
        # Verify the collection exists
        if not utility.has_collection(config.MILVUS_COLLECTION_NAME):
            print(f"Error: Collection '{config.MILVUS_COLLECTION_NAME}' does not exist.")
            # In a real app, you might exit or handle this more gracefully
            return
        
        # Load the collection into memory for faster searches
        print(f"Loading collection '{config.MILVUS_COLLECTION_NAME}' into memory...")
        collection = Collection(config.MILVUS_COLLECTION_NAME)
        collection.load()
        print("Collection loaded successfully.")
        
    except Exception as e:
        print(f"Could not connect to Milvus or load collection: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnects from Milvus on API shutdown."""
    print("Disconnecting from Milvus...")
    connections.disconnect("default")

# --- API Endpoint ---

@app.get("/similar_items/{item_id}", response_model=RecommendationResponse)
async def get_similar_items(item_id: str):
    """
    Finds items similar to the given item_id.

    - Fetches the vector for the source item_id.
    - Performs a vector similarity search in Milvus.
    - Returns the top 10 most similar items, excluding the source item itself.
    """
    try:
        collection = Collection(config.MILVUS_COLLECTION_NAME)

        # 1. Fetch the vector for the source item_id
        # We query the collection to get the embedding vector of our target item
        res = collection.query(
            expr=f"item_id == '{item_id}'",
            output_fields=["item_embedding"]
        )
        
        if not res:
            raise HTTPException(status_code=404, detail=f"Item with ID '{item_id}' not found.")
        
        source_vector = res[0]["item_embedding"]

        # 2. Perform similarity search
        search_params = {
            "metric_type": "L2",
            "params": {"ef": 128},
        }

        # Milvus returns k+1 results because the item itself is the most similar
        results = collection.search(
            data=[source_vector],
            anns_field="item_embedding",
            param=search_params,
            limit=11, # Ask for 11 to account for the item itself
            output_fields=["item_id"]
        )

        # 3. Format the response
        recommendations = []
        for hit in results[0]:
            # Exclude the source item from its own recommendation list
            if hit.entity.get('item_id') != item_id:
                recommendations.append(
                    SimilarItem(item_id=hit.entity.get('item_id'), distance=hit.distance)
                )
        
        # Ensure we only return up to 10 recommendations
        recommendations = recommendations[:10]

        return RecommendationResponse(source_item_id=item_id, recommendations=recommendations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")