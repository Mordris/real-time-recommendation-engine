# offline_pipeline/batch_embedder.py

import os
import pandas as pd
import requests
import gzip
import io
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sqlalchemy import create_engine
from tqdm import tqdm

# --- 1. Configuration ---
# Milvus Configuration
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'products'
DIMENSION = 384  # Dimension of 'all-MiniLM-L6-v2'

# PostgreSQL Configuration
POSTGRES_USER = 'admin'
POSTGRES_PASSWORD = 'admin'
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = '5432'
POSTGRES_DB = 'nebula_db'
TABLE_NAME = 'product_metadata'

# Data and Model Configuration
DATASET_URL = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz'
LOCAL_DATA_PATH = Path('./meta_Electronics.json.gz')
MODEL_NAME = 'all-MiniLM-L6-v2'
MAX_PRODUCTS = 10000  # Limit for the example

def download_data():
    """Downloads the dataset if it doesn't exist locally."""
    if LOCAL_DATA_PATH.exists():
        print(f"Dataset already exists at {LOCAL_DATA_PATH}. Skipping download.")
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    with requests.get(DATASET_URL, stream=True) as r:
        r.raise_for_status()
        with open(LOCAL_DATA_PATH, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading"):
                f.write(chunk)
    print("Download complete.")

def process_data():
    """Reads, parses, and cleans the dataset from the local gzipped file."""
    print("Processing dataset...")
    data = []
    with gzip.open(LOCAL_DATA_PATH, 'rt', encoding='utf-8') as gz:
        for line in tqdm(gz, desc="Processing JSON data"):
            try:
                # Using eval is generally unsafe, but acceptable for this trusted dataset.
                # A more robust solution for production would use a proper JSON parser that handles malformed lines.
                record = eval(line)
                data.append(record)
                if len(data) >= MAX_PRODUCTS:
                    break
            except (SyntaxError, NameError):
                continue

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} products.")

    # Data Cleaning
    df = df[['asin', 'title', 'description']].dropna()
    df = df.drop_duplicates(subset=['asin'])
    df['description'] = df['description'].str.strip()
    df = df[df['description'].str.len() > 20]
    df = df.head(MAX_PRODUCTS).reset_index(drop=True)
    # Important: Create a stable 'id' column from the new index for Milvus
    df['id'] = df.index
    print(f"Cleaned data, {len(df)} products remaining.")
    return df

def create_postgres_connection():
    """Creates a connection to the PostgreSQL database."""
    print("Waiting for PostgreSQL...")
    engine = None
    while not engine:
        try:
            engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')
            with engine.connect() as conn:
                print("PostgreSQL connection successful.")
                return engine
        except Exception as e:
            print(f"PostgreSQL not ready, retrying in 5 seconds... Error: {e}")
            time.sleep(5)


def load_metadata_to_postgres(df, engine):
    """Loads product metadata into PostgreSQL."""
    print("Loading metadata to PostgreSQL...")
    metadata_df = df[['id', 'asin', 'title', 'description']]
    metadata_df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
    print(f"Successfully loaded {len(metadata_df)} records into PostgreSQL table '{TABLE_NAME}'.")

def create_milvus_connection():
    """Connects to Milvus with retries."""
    print("Waiting for Milvus...")
    while True:
        try:
            connections.connect('default', host=MILVUS_HOST, port=MILVUS_PORT)
            if utility.has_collection(COLLECTION_NAME):
                print(f"Found existing collection '{COLLECTION_NAME}'.")
            print("Milvus connection successful.")
            return
        except Exception as e:
            print(f"Milvus not ready, retrying in 5 seconds... Error: {e}")
            time.sleep(5)

def create_milvus_collection():
    """Creates a new collection in Milvus, dropping it if it exists."""
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Dropping it.")
        utility.drop_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True),
        FieldSchema(name='asin', dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields, description='Product Embeddings')
    collection = Collection(name=COLLECTION_NAME, schema=schema, using='default')

    print("Creating IVF_FLAT index on 'embedding' field...")
    index_params = {
        'metric_type': 'L2',
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 128}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    print("Milvus collection and index setup complete.")
    return collection

def generate_and_insert_embeddings(df, collection):
    """Generates embeddings and inserts them into Milvus."""
    print(f"Loading sentence transformer model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings and inserting into Milvus in batches...")
    batch_size = 500
    for i in tqdm(range(0, len(df), batch_size), desc="Embedding and Inserting"):
        batch_df = df.iloc[i:i+batch_size]
        text_to_embed = (batch_df['title'] + ". " + batch_df['description']).tolist()
        embeddings = model.encode(text_to_embed, show_progress_bar=False)

        data = [
            batch_df['id'].tolist(),
            batch_df['asin'].tolist(),
            embeddings
        ]
        collection.insert(data)

    collection.flush()
    print(f"\nSuccessfully inserted {len(df)} embeddings into Milvus.")
    print(f"Total entities in collection: {collection.num_entities}")
    collection.load()
    print("Collection loaded into memory for searching.")

if __name__ == "__main__":
    # 0. Download data if needed
    download_data()

    # 1. Process and clean data
    product_df = process_data()

    # 2. Setup PostgreSQL
    pg_engine = create_postgres_connection()
    load_metadata_to_postgres(product_df, pg_engine)

    # 3. Setup Milvus
    create_milvus_connection()
    milvus_collection = create_milvus_collection()

    # 4. Generate and insert embeddings
    generate_and_insert_embeddings(product_df, milvus_collection)

    print("\nOffline pipeline finished successfully!")
    connections.disconnect('default')