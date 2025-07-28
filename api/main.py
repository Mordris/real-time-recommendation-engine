# /api/main.py
# Main FastAPI application for the Nebula Recommendation Engine.

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Literal
import json
import logging

from pymilvus import connections, utility, Collection
from kafka import KafkaProducer
from kafka.errors import KafkaError

import sys
sys.path.append('..')
import config

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class SimilarItem(BaseModel):
    item_id: str = Field(..., description="The unique ID of the similar item (ASIN).")
    distance: float = Field(..., description="The similarity score (L2 distance). Lower is more similar.")

class RecommendationResponse(BaseModel):
    source_item_id: str = Field(..., description="The item ID provided in the request.")
    recommendations: List[SimilarItem]

class UserInteraction(BaseModel):
    user_id: str = Field(..., description="The ID of the user performing the action.")
    item_id: str = Field(..., description="The ID of the item being interacted with.")
    event_type: Literal['click', 'view', 'add_to_cart'] = Field(..., description="The type of interaction.")

class InteractionResponse(BaseModel):
    status: str
    message: str

# --- FastAPI Application Setup ---

app = FastAPI(
    title="Nebula Recommendation Engine API",
    description="Provides content-based item similarity recommendations and ingests user interactions.",
    version="0.2.0"
)

# --- Service Connection Management (Milvus & Kafka) ---

@app.on_event("startup")
async def startup_event():
    # Store connections in the app state
    app.state.milvus_collection = None
    app.state.kafka_producer = None

    # Connect to Milvus
    try:
        logger.info("Connecting to Milvus...")
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        
        if not utility.has_collection(config.MILVUS_COLLECTION_NAME):
            logger.error(f"FATAL: Milvus collection '{config.MILVUS_COLLECTION_NAME}' does not exist.")
            return
            
        collection = Collection(config.MILVUS_COLLECTION_NAME)
        collection.load()
        app.state.milvus_collection = collection
        logger.info("Successfully connected to Milvus and loaded collection.")

    except Exception as e:
        logger.error(f"Could not connect to Milvus: {e}")

    # Connect to Kafka (Redpanda)
    try:
        logger.info(f"Connecting to Kafka producer at {config.KAFKA_BOOTSTRAP_SERVERS}...")
        producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=3, # Retry up to 3 times on failure
            acks='all' # Wait for all in-sync replicas to acknowledge
        )
        app.state.kafka_producer = producer
        logger.info("Successfully connected to Kafka producer.")
    except Exception as e:
        logger.error(f"Could not connect to Kafka producer: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if app.state.kafka_producer:
        logger.info("Closing Kafka producer...")
        app.state.kafka_producer.flush()
        app.state.kafka_producer.close()
    
    logger.info("Disconnecting from Milvus...")
    connections.disconnect("default")

# --- API Endpoints ---

@app.post("/interaction", response_model=InteractionResponse)
async def post_interaction(interaction: UserInteraction, request: Request):
    """
    Accepts a user interaction event and produces it to a Kafka topic.
    """
    producer = request.app.state.kafka_producer
    if not producer:
        raise HTTPException(status_code=503, detail="Kafka producer is not available.")

    try:
        # The key is the user_id, ensuring all events for a user go to the same partition
        key = interaction.user_id.encode('utf-8')
        
        future = producer.send(
            config.USER_INTERACTIONS_TOPIC, 
            value=interaction.dict(),
            key=key
        )
        # Block for 'successful' send; raises an exception on failure.
        future.get(timeout=5) 
        logger.info(f"Successfully produced event for user '{interaction.user_id}' to topic '{config.USER_INTERACTIONS_TOPIC}'.")
        return InteractionResponse(status="ok", message="Interaction event received.")
    except KafkaError as e:
        logger.error(f"Failed to send message to Kafka: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to produce event to messaging system: {e}")

@app.get("/similar_items/{item_id}", response_model=RecommendationResponse)
async def get_similar_items(item_id: str, request: Request):
    """
    Finds items similar to the given item_id.
    """
    collection = request.app.state.milvus_collection
    if not collection:
        raise HTTPException(status_code=503, detail="Milvus collection is not available.")
        
    try:
        res = collection.query(expr=f"item_id == '{item_id}'", output_fields=["item_embedding"])
        
        if not res:
            raise HTTPException(status_code=404, detail=f"Item with ID '{item_id}' not found.")
        
        source_vector = res[0]["item_embedding"]
        
        search_params = {"metric_type": "L2", "params": {"ef": 128}}
        
        results = collection.search(
            data=[source_vector],
            anns_field="item_embedding",
            param=search_params,
            limit=11,
            output_fields=["item_id"]
        )
        
        recommendations = [
            SimilarItem(item_id=hit.entity.get('item_id'), distance=hit.distance)
            for hit in results[0] if hit.entity.get('item_id') != item_id
        ][:10]
        
        return RecommendationResponse(source_item_id=item_id, recommendations=recommendations)

    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")