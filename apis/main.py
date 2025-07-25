# apis/main.py

import os
import json
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymilvus import connections, utility, Collection
from sqlalchemy import create_engine, text
from kafka import KafkaProducer
from kafka.errors import KafkaError

# --- 0. Configuration ---
logging.basicConfig(level=logging.INFO)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "products"

POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "nebula_db")
TABLE_NAME = "product_metadata"

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INTERACTIONS_TOPIC = os.getenv("INTERACTIONS_TOPIC", "user_interactions")

TOP_K = 10

# --- 1. Pydantic Models for Data Validation ---
class UserInteraction(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    item_id: int = Field(..., description="The unique identifier for the item (the primary key in Milvus).")
    event_type: str = Field(..., description="The type of interaction (e.g., 'click', 'view', 'purchase').")

# --- 2. Initialize FastAPI App ---
app = FastAPI(
    title="Nebula Recommendation Engine API",
    description="API for event ingestion and serving recommendations.",
    version="1.3.0" # Version bump
)

# --- 3. Database and Service Connections ---
db_engine = None
milvus_collection = None
kafka_producer = None

@app.on_event("startup")
def startup_event():
    global db_engine, milvus_collection, kafka_producer
    
    # Connect to PostgreSQL
    try:
        logging.info("Connecting to PostgreSQL...")
        db_url = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
        db_engine = create_engine(db_url)
        with db_engine.connect() as conn:
            logging.info("PostgreSQL connection successful.")
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL: {e}")
        
    # --- UPDATED SECTION: Connect to Milvus with a longer timeout and retries ---
    max_retries = 10
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            logging.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT} (Attempt {attempt + 1}/{max_retries})...")
            # NEW: Increased connection timeout to 20 seconds
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=20)
            
            if utility.has_collection(COLLECTION_NAME):
                milvus_collection = Collection(COLLECTION_NAME)
                milvus_collection.load()
                logging.info("Milvus connection successful and collection loaded.")
            else:
                logging.warning(f"Milvus connection successful, but collection '{COLLECTION_NAME}' does not exist yet.")
            
            break
        except Exception as e:
            logging.error(f"Error connecting to Milvus on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Could not connect to Milvus after several attempts. API will start but may be unhealthy.")
    # --- END UPDATED SECTION ---

    # Connect to Kafka (Redpanda)
    try:
        logging.info(f"Connecting to Kafka producer at {KAFKA_BOOTSTRAP_SERVERS}...")
        kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logging.info("Kafka producer connection successful.")
    except KafkaError as e:
        logging.error(f"Error connecting to Kafka: {e}")

@app.on_event("shutdown")
def shutdown_event():
    logging.info("Disconnecting from Milvus...")
    connections.disconnect("default")
    if kafka_producer:
        logging.info("Closing Kafka producer...")
        kafka_producer.close()

# --- 4. API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Nebula API is running"}

@app.post("/interaction", status_code=202)
async def log_interaction(interaction: UserInteraction):
    if kafka_producer is None:
        raise HTTPException(status_code=503, detail="Kafka producer is not available.")
    
    try:
        future = kafka_producer.send(INTERACTIONS_TOPIC, value=interaction.dict())
        record_metadata = future.get(timeout=10)
        logging.info(f"Message sent to topic '{record_metadata.topic}' at partition {record_metadata.partition}")
        return {"status": "interaction accepted", "topic": record_metadata.topic}
    except KafkaError as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message to Kafka: {e}")

@app.get("/similar_items/{item_id}")
def get_similar_items(item_id: int):
    global milvus_collection
    if milvus_collection is None:
        if utility.has_collection(COLLECTION_NAME):
            milvus_collection = Collection(COLLECTION_NAME)
            milvus_collection.load()
        else:
             raise HTTPException(status_code=503, detail="Milvus collection not loaded. Has the offline pipeline been run?")

    if db_engine is None:
        raise HTTPException(status_code=503, detail="PostgreSQL service is not available.")
        
    try:
        results = milvus_collection.query(expr=f"id == {item_id}", output_fields=["embedding", "asin"])
        if not results:
            raise HTTPException(status_code=404, detail=f"Item with ID {item_id} not found.")
        
        target_vector = results[0]['embedding']
        query_asin = results[0]['asin']

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        search_results = milvus_collection.search(
            data=[target_vector],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K + 1,
            output_fields=["asin"]
        )
        
        similar_item_asins = [
            hit.entity.get('asin') for hit in search_results[0] if hit.entity.get('asin') != query_asin
        ][:TOP_K]
        
        if not similar_item_asins:
            return {"message": "No similar items found."}

        with db_engine.connect() as conn:
            query = text(f"SELECT asin, title, description FROM {TABLE_NAME} WHERE asin IN :asins")
            metadata_results = conn.execute(query, {"asins": tuple(similar_item_asins)}).fetchall()

        response_items = [
            {"asin": row[0], "title": row[1], "description": row[2]} for row in metadata_results
        ]
        
        return {"query_item_id": item_id, "similar_items": response_items}

    except Exception as e:
        logging.error(f"An error occurred during similarity search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during similarity search.")