# batch_embedder.py
# This script performs the offline batch processing for the Nebula project.
# It reads product data, generates vector embeddings, and inserts them into Milvus.

import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import gzip
import ast

import config

def load_dataset_from_json_gz(path):
    """
    Loads and parses a gzipped file containing Python dict literals into a pandas DataFrame.
    """
    data = []
    print(f"Loading and parsing gzipped dataset from '{path}'...")
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading file lines"):
                try:
                    item = ast.literal_eval(line)
                    record = {
                        config.ITEM_ID_COLUMN: item.get(config.ITEM_ID_COLUMN),
                        config.ITEM_TEXT_COLUMN_1: item.get(config.ITEM_TEXT_COLUMN_1),
                        config.ITEM_TEXT_COLUMN_2: item.get(config.ITEM_TEXT_COLUMN_2, '')
                    }
                    data.append(record)
                except (ValueError, SyntaxError, AttributeError):
                    continue
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{path}'.")
        print("Please ensure you have downloaded the dataset and placed it correctly.")
        return None

def create_milvus_collection():
    """
    Connects to Milvus and sets up the collection with the required schema.
    If the collection already exists, it will be dropped and recreated.
    """
    print("Attempting to connect to Milvus...")
    connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    print(f"Successfully connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")

    if utility.has_collection(config.MILVUS_COLLECTION_NAME):
        print(f"Collection '{config.MILVUS_COLLECTION_NAME}' already exists. Dropping it.")
        utility.drop_collection(config.MILVUS_COLLECTION_NAME)

    # Increased max_length to a very safe value.
    fields = [
        FieldSchema(name="item_id", dtype=DataType.VARCHAR, is_primary=True, max_length=50),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1500), 
        FieldSchema(name="item_embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields, "Nebula item collection for recommendations")
    
    print(f"Creating collection '{config.MILVUS_COLLECTION_NAME}'...")
    collection = Collection(config.MILVUS_COLLECTION_NAME, schema, consistency_level="Strong")
    print("Collection created successfully.")
    return collection

def create_vector_index(collection):
    """
    Creates a vector index on the 'item_embedding' field for efficient searching.
    """
    print("Creating index for the 'item_embedding' field...")
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200},
    }
    collection.create_index("item_embedding", index_params)
    print("Index created successfully.")

def main():
    """
    Main function to orchestrate the data loading, embedding, and insertion process.
    """
    collection = create_milvus_collection()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading sentence transformer model '{config.EMBEDDING_MODEL}' onto device: {device}")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    print("Model loaded successfully.")

    df = load_dataset_from_json_gz(config.DATASET_PATH)
    if df is None:
        return

    df.dropna(subset=[config.ITEM_ID_COLUMN, config.ITEM_TEXT_COLUMN_1], inplace=True)
    if config.ITEM_TEXT_COLUMN_2 not in df.columns:
        df[config.ITEM_TEXT_COLUMN_2] = ''
    df[config.ITEM_TEXT_COLUMN_2] = df[config.ITEM_TEXT_COLUMN_2].fillna('')
    
    print(f"Dataset loaded and preprocessed. Found {len(df)} items to embed.")

    print(f"Starting data embedding and insertion in batches of {config.BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(df), config.BATCH_SIZE), desc="Processing Batches"):
        batch = df.iloc[i:i + config.BATCH_SIZE]

        text_to_embed = (batch[config.ITEM_TEXT_COLUMN_1].str.slice(0, 256) + " " + batch[config.ITEM_TEXT_COLUMN_2].str.slice(0, 512)).tolist()
        
        embeddings = model.encode(text_to_embed, show_progress_bar=False)

        # *** THE FIX IS HERE ***
        # Truncate the title to the max length allowed by the schema before inserting.
        truncated_titles = batch[config.ITEM_TEXT_COLUMN_1].str.slice(0, 1500).tolist()

        entities = [
            batch[config.ITEM_ID_COLUMN].tolist(),
            truncated_titles, # Use the truncated titles
            embeddings
        ]
        
        try:
            collection.insert(entities)
        except Exception as e:
            print(f"An error occurred during insertion on batch starting at index {i}: {e}")
            continue

    print("All batches inserted. Flushing data to make it searchable...")
    collection.flush()
    print("Data flushed.")
    
    create_vector_index(collection)
    
    print("Loading collection into memory for querying...")
    collection.load()
    
    print(f"Process complete. Final entity count in collection: {collection.num_entities}")

if __name__ == "__main__":
    main()