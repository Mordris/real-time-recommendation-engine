# config.py
# Central configuration file for the Nebula project.

# --- Milvus Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION_NAME = "nebula_items"

# --- Redpanda/Kafka Configuration ---
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
USER_INTERACTIONS_TOPIC = "user_interactions"

# --- Embedding Model Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Data File Configuration (Updated for SNAP Electronics Dataset) ---
DATASET_PATH = "data/meta_Electronics.json.gz"
ITEM_ID_COLUMN = "asin"
ITEM_TEXT_COLUMN_1 = "title"
ITEM_TEXT_COLUMN_2 = "description"

# --- Batch Processing Configuration ---
BATCH_SIZE = 128