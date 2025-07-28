# config.py
# Central configuration file for the Nebula project.

# --- Milvus Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION_NAME = "nebula_items"

# --- Embedding Model Configuration ---
# We use all-MiniLM-L6-v2, which produces 384-dimensional vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Data File Configuration (Updated for SNAP Electronics Dataset) ---
DATASET_PATH = "data/meta_Electronics.json.gz"
ITEM_ID_COLUMN = "asin"         # Changed from 'product_id'
ITEM_TEXT_COLUMN_1 = "title"    # Changed from 'Title'
ITEM_TEXT_COLUMN_2 = "description" # Changed from 'Description'

# --- Batch Processing Configuration ---
BATCH_SIZE = 128 # How many items to process and insert at once