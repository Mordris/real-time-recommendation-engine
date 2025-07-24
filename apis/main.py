import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymilvus import connections, utility, Collection
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError

# --- 0. Configuration ---
load_dotenv()

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "products"

# PostgreSQL Configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "nebula_db")
TABLE_NAME = "product_metadata"

# Redpanda/Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INTERACTIONS_TOPIC = "user_interactions"

# Search Configuration
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
    version="1.1.0"
)

# --- 3. Database and Service Connections ---
db_engine = create_engine(
    f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
)

milvus_collection = None
kafka_producer = None

@app.on_event("startup")
def startup_event():
    global milvus_collection, kafka_producer
    # Connect to Milvus
    try:
        print("Connecting to Milvus...")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Milvus collection '{COLLECTION_NAME}' does not exist.")
        milvus_collection = Collection(COLLECTION_NAME)
        milvus_collection.load()
        print("Milvus connection successful and collection loaded.")
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        raise RuntimeError("Could not connect to Milvus.") from e

    # Connect to Kafka (Redpanda)
    try:
        print("Connecting to Kafka producer...")
        kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("Kafka producer connection successful.")
    except KafkaError as e:
        print(f"Error connecting to Kafka: {e}")
        raise RuntimeError("Could not connect to Kafka producer.") from e

@app.on_event("shutdown")
def shutdown_event():
    print("Disconnecting from Milvus...")
    connections.disconnect("default")
    if kafka_producer:
        print("Closing Kafka producer...")
        kafka_producer.close()

# --- 4. API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Nebula API is running"}

@app.post("/interaction", status_code=202)
def log_interaction(interaction: UserInteraction):
    """
    Logs a user interaction event by sending it to a Kafka topic.
    """
    if kafka_producer is None:
        raise HTTPException(status_code=503, detail="Kafka producer is not available.")
    
    try:
        future = kafka_producer.send(INTERACTIONS_TOPIC, value=interaction.dict())
        # Block for 'successful' sends.
        record_metadata = future.get(timeout=10)
        print(f"Message sent to topic '{record_metadata.topic}' at partition {record_metadata.partition}")
        return {"status": "interaction accepted", "topic": record_metadata.topic}
    except KafkaError as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message to Kafka: {e}")


@app.get("/similar_items/{item_id}")
def get_similar_items(item_id: int):
    """
    Finds items similar to a given item ID.
    - item_id: The primary key ID of the item in the Milvus collection (e.g., from 0 to 8339).
    """
    if milvus_collection is None:
        raise HTTPException(status_code=503, detail="Milvus service is not available.")

    try:
        results = milvus_collection.query(
            expr=f"id == {item_id}",
            output_fields=["embedding"]
        )
        if not results:
            raise HTTPException(status_code=404, detail=f"Item with ID {item_id} not found.")
        
        target_vector = results[0]['embedding']

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        search_results = milvus_collection.search(
            data=[target_vector],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K + 1,
            output_fields=["asin"]
        )
        
        similar_item_asins = []
        for hit in search_results[0]:
            if hit.id != item_id:
                similar_item_asins.append(hit.entity.get('asin'))
        
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
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during similarity search.")