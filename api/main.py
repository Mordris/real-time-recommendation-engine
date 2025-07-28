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
sys.path.append('..')
import config

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class SimilarItem(BaseModel):
    item_id: str = Field(..., description="The unique ID of the similar item (ASIN).")
    distance: float = Field(..., description="The similarity score (L2 distance). Lower is more similar.")

class StaticRecommendationResponse(BaseModel):
    source_item_id: str = Field(..., description="The item ID provided in the request.")
    recommendations: List[SimilarItem]

class RealtimeRecommendationResponse(BaseModel):
    user_id: str = Field(..., description="The user ID for whom recommendations were generated.")
    recommendations: List[str] = Field(..., description="A list of recommended item IDs.")

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
    description="Provides real-time and static recommendations, and ingests user interactions.",
    version="0.3.0"
)

# --- Service Connection Management ---
@app.on_event("startup")
async def startup_event():
    app.state.milvus_collection = None
    app.state.kafka_producer = None
    app.state.redis_client = None

    # Connect to Milvus
    try:
        logger.info("Connecting to Milvus...")
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        if utility.has_collection(config.MILVUS_COLLECTION_NAME):
            collection = Collection(config.MILVUS_COLLECTION_NAME)
            collection.load()
            app.state.milvus_collection = collection
            logger.info("Successfully connected to Milvus and loaded collection.")
        else:
            logger.error(f"FATAL: Milvus collection '{config.MILVUS_COLLECTION_NAME}' does not exist.")
    except Exception as e:
        logger.error(f"Could not connect to Milvus: {e}")

    # Connect to Kafka
    try:
        logger.info(f"Connecting to Kafka producer at {config.KAFKA_BOOTSTRAP_SERVERS}...")
        app.state.kafka_producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=3, acks='all'
        )
        logger.info("Successfully connected to Kafka producer.")
    except Exception as e:
        logger.error(f"Could not connect to Kafka producer: {e}")

    # Connect to Redis
    try:
        logger.info(f"Connecting to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}...")
        app.state.redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
        app.state.redis_client.ping()
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.error(f"Could not connect to Redis: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    if app.state.kafka_producer:
        logger.info("Closing Kafka producer...")
        app.state.kafka_producer.flush()
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
    """
    redis_client = request.app.state.redis_client
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis client is not available.")
    
    redis_key = f"{config.RECOMMENDATION_KEY_PREFIX}{user_id}"
    cached_recs = redis_client.get(redis_key)
    
    recommendations = json.loads(cached_recs) if cached_recs else []
    
    return RealtimeRecommendationResponse(user_id=user_id, recommendations=recommendations)


@app.post("/interaction", response_model=InteractionResponse)
async def post_interaction(interaction: UserInteraction, request: Request):
    """
    Accepts a user interaction event and produces it to a Kafka topic.
    """
    producer = request.app.state.kafka_producer
    if not producer:
        raise HTTPException(status_code=503, detail="Kafka producer is not available.")
    try:
        key = interaction.user_id.encode('utf-8')
        future = producer.send(config.USER_INTERACTIONS_TOPIC, value=interaction.dict(), key=key)
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
    """
    collection = request.app.state.milvus_collection
    if not collection:
        raise HTTPException(status_code=503, detail="Milvus collection is not available.")
    try:
        res = collection.query(expr=f"item_id == '{item_id}'", output_fields=["item_embedding"])
        if not res:
            raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")
        
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
        
        return StaticRecommendationResponse(source_item_id=item_id, recommendations=recommendations)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))