# /api/main.py
# Main FastAPI application for the Nebula Recommendation Engine.

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Literal
import json
import logging
import redis

from pymilvus import connections, utility, Collection
from kafka import KafkaProducer
from kafka.errors import KafkaError

import sys
# Add the project root to the Python path to allow importing the central 'config' module.
sys.path.append('..')
import config

# --- Setup Logging ---
# Configure basic logging to show informational messages.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
# Pydantic models define the data structures for API requests and responses.
# FastAPI uses them to perform automatic data validation, serialization, and documentation.

class SimilarItem(BaseModel):
    """Defines the structure for a single similar item in a static recommendation response."""
    item_id: str = Field(..., description="The unique ID of the similar item (ASIN).")
    distance: float = Field(..., description="The similarity score (L2 distance). Lower is more similar.")

class StaticRecommendationResponse(BaseModel):
    """Defines the response structure for the static /similar_items endpoint."""
    source_item_id: str = Field(..., description="The item ID provided in the request.")
    recommendations: List[SimilarItem]

class RealtimeRecommendationResponse(BaseModel):
    """Defines the response structure for the real-time /recommendations endpoint."""
    user_id: str = Field(..., description="The user ID for whom recommendations were generated.")
    recommendations: List[str] = Field(..., description="A list of recommended item IDs.")

class UserInteraction(BaseModel):
    """Defines the input structure for a user interaction event sent to the /interaction endpoint."""
    user_id: str = Field(..., description="The ID of the user performing the action.")
    item_id: str = Field(..., description="The ID of the item being interacted with.")
    event_type: Literal['click', 'view', 'add_to_cart'] = Field(..., description="The type of interaction.")

class InteractionResponse(BaseModel):
    """Defines the simple success response for the /interaction endpoint."""
    status: str
    message: str

# --- FastAPI Application Setup ---
# Initialize the main FastAPI application instance.
app = FastAPI(
    title="Nebula Recommendation Engine API",
    description="Provides real-time and static recommendations, and ingests user interactions.",
    version="0.3.0"
)

# --- Service Connection Management ---
# FastAPI startup and shutdown events are used to manage the lifecycle of external service connections.
# This is a best practice to avoid the overhead of connecting and disconnecting on every API request.

@app.on_event("startup")
async def startup_event():
    """
    Initializes and manages connections to Milvus, Kafka, and Redis when the API server starts.
    The connection objects are stored in the application's state for access in endpoints.
    """
    # Initialize state variables to None.
    app.state.milvus_collection = None
    app.state.kafka_producer = None
    app.state.redis_client = None

    # Connect to Milvus vector database.
    try:
        logger.info("Connecting to Milvus...")
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        if utility.has_collection(config.MILVUS_COLLECTION_NAME):
            collection = Collection(config.MILVUS_COLLECTION_NAME)
            collection.load() # Pre-load the collection into memory for fast search performance.
            app.state.milvus_collection = collection
            logger.info("Successfully connected to Milvus and loaded collection.")
        else:
            logger.error(f"FATAL: Milvus collection '{config.MILVUS_COLLECTION_NAME}' does not exist.")
    except Exception as e:
        logger.error(f"Could not connect to Milvus: {e}")

    # Connect to Kafka (Redpanda) as a producer.
    try:
        logger.info(f"Connecting to Kafka producer at {config.KAFKA_BOOTSTRAP_SERVERS}...")
        app.state.kafka_producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            # Serialize message values as JSON strings encoded in UTF-8.
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=3, # Retry sending a message up to 3 times on transient errors.
            acks='all' # Wait for acknowledgment from all in-sync replicas for high durability.
        )
        logger.info("Successfully connected to Kafka producer.")
    except Exception as e:
        logger.error(f"Could not connect to Kafka producer: {e}")

    # Connect to Redis cache.
    try:
        logger.info(f"Connecting to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}...")
        # decode_responses=True ensures that Redis returns strings, not bytes.
        app.state.redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
        app.state.redis_client.ping() # Verify the connection is active.
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.error(f"Could not connect to Redis: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Gracefully closes all external service connections when the API server shuts down.
    """
    if app.state.kafka_producer:
        logger.info("Closing Kafka producer...")
        app.state.kafka_producer.flush() # Ensure all buffered messages are sent.
        app.state.kafka_producer.close()
    if app.state.redis_client:
        logger.info("Closing Redis connection...")
        app.state.redis_client.close()
    logger.info("Disconnecting from Milvus...")
    connections.disconnect("default")


# --- API Endpoints ---

@app.get("/recommendations/{user_id}", response_model=RealtimeRecommendationResponse)
async def get_realtime_recommendations(user_id: str, request: Request):
    """
    Fetches the latest real-time recommendations for a user from the Redis cache.
    This endpoint is designed to be extremely fast as it only performs a key-value lookup.
    """
    redis_client = request.app.state.redis_client
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis client is not available.")
    
    # Construct the key where the user's recommendations are stored.
    redis_key = f"{config.RECOMMENDATION_KEY_PREFIX}{user_id}"
    cached_recs = redis_client.get(redis_key)
    
    # If recommendations exist in the cache, parse the JSON string; otherwise, return an empty list.
    recommendations = json.loads(cached_recs) if cached_recs else []
    
    return RealtimeRecommendationResponse(user_id=user_id, recommendations=recommendations)


@app.post("/interaction", response_model=InteractionResponse)
async def post_interaction(interaction: UserInteraction, request: Request):
    """
    Accepts a user interaction event and produces it to a Kafka topic for asynchronous processing.
    """
    producer = request.app.state.kafka_producer
    if not producer:
        raise HTTPException(status_code=503, detail="Kafka producer is not available.")
    try:
        # Use the user_id as the message key. This ensures that all events for the same user
        # are sent to the same Kafka partition, which is crucial for stateful processing.
        key = interaction.user_id.encode('utf-8')
        
        # Send the event to the configured Kafka topic.
        future = producer.send(config.USER_INTERACTIONS_TOPIC, value=interaction.dict(), key=key)
        
        # Block for a short timeout to confirm the message was sent successfully.
        future.get(timeout=5) 
        
        logger.info(f"Successfully produced event for user '{interaction.user_id}'.")
        return InteractionResponse(status="ok", message="Interaction event received.")
    except KafkaError as e:
        logger.error(f"Failed to send message to Kafka: {e}")
        raise HTTPException(status_code=500, detail="Failed to produce event.")

@app.get("/similar_items/{item_id}", response_model=StaticRecommendationResponse)
async def get_similar_items(item_id: str, request: Request):
    """
    Finds items similar to the given item_id (static content-based recommendations).
    This performs a live query against the Milvus database.
    """
    collection = request.app.state.milvus_collection
    if not collection:
        raise HTTPException(status_code=503, detail="Milvus collection is not available.")
    try:
        # Step 1: Fetch the vector for the source item from Milvus.
        res = collection.query(expr=f"item_id == '{item_id}'", output_fields=["item_embedding"])
        
        # Handle case where the item ID does not exist.
        if not res:
            raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")
        
        source_vector = res[0]["item_embedding"]
        
        # Step 2: Use the source vector to perform a similarity search.
        search_params = {"metric_type": "L2", "params": {"ef": 128}}
        results = collection.search(
            data=[source_vector],
            anns_field="item_embedding",
            param=search_params,
            limit=11, # Ask for 11 results to account for the item itself being the most similar.
            output_fields=["item_id"]
        )
        
        # Step 3: Format the results, excluding the source item from its own recommendation list.
        recommendations = [
            SimilarItem(item_id=hit.entity.get('item_id'), distance=hit.distance)
            for hit in results[0] if hit.entity.get('item_id') != item_id
        ][:10] # Ensure we only return a maximum of 10 items.
        
        return StaticRecommendationResponse(source_item_id=item_id, recommendations=recommendations)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))