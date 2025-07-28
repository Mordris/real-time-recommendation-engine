# /batch_embedder.py
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
import ast # Abstract Syntax Tree module used for safely parsing Python literals from strings.

import config

def load_dataset_from_json_gz(path):
    """
    Loads and parses a gzipped file containing Python dict literals into a pandas DataFrame.
    The Stanford SNAP dataset uses single quotes for strings, which is valid Python syntax but
    not valid JSON. Therefore, we use `ast.literal_eval` instead of `json.loads`.
    """
    data = []
    print(f"Loading and parsing gzipped dataset from '{path}'...")
    try:
        # Open the gzipped file in text mode.
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            # `tqdm` provides a progress bar for long-running loops.
            for line in tqdm(f, desc="Reading file lines"):
                try:
                    # Safely evaluate the string as a Python literal (handles single quotes).
                    item = ast.literal_eval(line)
                    # Build a new dictionary with only the fields we need, using .get()
                    # to prevent errors if a key is missing in a particular line.
                    record = {
                        config.ITEM_ID_COLUMN: item.get(config.ITEM_ID_COLUMN),
                        config.ITEM_TEXT_COLUMN_1: item.get(config.ITEM_TEXT_COLUMN_1),
                        config.ITEM_TEXT_COLUMN_2: item.get(config.ITEM_TEXT_COLUMN_2, '') # Default to empty string
                    }
                    data.append(record)
                except (ValueError, SyntaxError, AttributeError):
                    # If a line is malformed, skip it and continue.
                    continue
        # Convert the list of dictionaries into a pandas DataFrame for easier processing.
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{path}'.")
        print("Please ensure you have downloaded the dataset and placed it correctly.")
        return None

def create_milvus_collection():
    """
    Connects to Milvus and sets up the collection with the required schema.
    If the collection already exists, it will be dropped and recreated to ensure a clean slate.
    """
    print("Attempting to connect to Milvus...")
    connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    print(f"Successfully connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")

    # Check if the collection already exists from a previous run.
    if utility.has_collection(config.MILVUS_COLLECTION_NAME):
        print(f"Collection '{config.MILVUS_COLLECTION_NAME}' already exists. Dropping it.")
        utility.drop_collection(config.MILVUS_COLLECTION_NAME)

    # Define the schema for our collection. This tells Milvus what our data will look like.
    fields = [
        # The primary key field. Must be unique for each entity.
        FieldSchema(name="item_id", dtype=DataType.VARCHAR, is_primary=True, max_length=50),
        # A field to store the product title for potential future use (e.g., display in a UI).
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1500), 
        # The main vector field where we'll store the embeddings.
        FieldSchema(name="item_embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields, "Nebula item collection for recommendations")
    
    print(f"Creating collection '{config.MILVUS_COLLECTION_NAME}'...")
    # Create the collection object with a strong consistency level to ensure writes are immediately readable.
    collection = Collection(config.MILVUS_COLLECTION_NAME, schema, consistency_level="Strong")
    print("Collection created successfully.")
    return collection

def create_vector_index(collection):
    """
    Creates a vector index on the 'item_embedding' field. An index is crucial for
    performing fast and efficient similarity searches. Without an index, Milvus would have
    to compare a query vector to every single vector in the database (brute-force search).
    """
    print("Creating index for the 'item_embedding' field...")
    index_params = {
        "metric_type": "L2",       # L2 (Euclidean) distance is a good choice for Sentence Transformer embeddings.
        "index_type": "HNSW",      # HNSW is a high-performance, graph-based index known for its speed and accuracy.
        "params": {"M": 8, "efConstruction": 200}, # Index-specific parameters tuning the build process.
    }
    collection.create_index("item_embedding", index_params)
    print("Index created successfully.")

def main():
    """
    Main function to orchestrate the data loading, embedding, and insertion process.
    """
    # Step 1: Initialize the Milvus collection.
    collection = create_milvus_collection()

    # Step 2: Load the pre-trained Sentence Transformer model from Hugging Face.
    # Check for CUDA availability and use the GPU if possible for a massive speedup.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading sentence transformer model '{config.EMBEDDING_MODEL}' onto device: {device}")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    print("Model loaded successfully.")

    # Step 3: Load the dataset from the gzipped file into a DataFrame.
    df = load_dataset_from_json_gz(config.DATASET_PATH)
    if df is None:
        # Exit if the file was not found.
        return

    # Step 4: Preprocess the DataFrame.
    # Drop any rows that are missing an ID or a title, as they are essential.
    df.dropna(subset=[config.ITEM_ID_COLUMN, config.ITEM_TEXT_COLUMN_1], inplace=True)
    # Ensure the description column exists and fill any missing descriptions with an empty string.
    if config.ITEM_TEXT_COLUMN_2 not in df.columns:
        df[config.ITEM_TEXT_COLUMN_2] = ''
    df[config.ITEM_TEXT_COLUMN_2] = df[config.ITEM_TEXT_COLUMN_2].fillna('')
    
    print(f"Dataset loaded and preprocessed. Found {len(df)} items to embed.")

    # Step 5: Process the data in batches to manage memory usage.
    print(f"Starting data embedding and insertion in batches of {config.BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(df), config.BATCH_SIZE), desc="Processing Batches"):
        batch = df.iloc[i:i + config.BATCH_SIZE]

        # Combine title and description to create a richer source text for the embedding model.
        # We slice the text to prevent extremely long inputs, which can slow down the model.
        text_to_embed = (batch[config.ITEM_TEXT_COLUMN_1].str.slice(0, 256) + " " + batch[config.ITEM_TEXT_COLUMN_2].str.slice(0, 512)).tolist()
        
        # Generate the vector embeddings for the current batch. This is the most compute-intensive step.
        embeddings = model.encode(text_to_embed, show_progress_bar=False)

        # Truncate titles to ensure they fit within the Milvus schema's max_length.
        # This prevents `ParamError` exceptions during insertion.
        truncated_titles = batch[config.ITEM_TEXT_COLUMN_1].str.slice(0, 1500).tolist()

        # Prepare the data in the format Milvus expects: a list of lists, where each inner list
        # corresponds to a field in the schema (item_id, title, item_embedding).
        entities = [
            batch[config.ITEM_ID_COLUMN].tolist(),
            truncated_titles, # Use the truncated titles
            embeddings
        ]
        
        try:
            # Insert the batch of entities into the Milvus collection.
            collection.insert(entities)
        except Exception as e:
            # If an error occurs (e.g., a network issue), log it and continue with the next batch.
            print(f"An error occurred during insertion on batch starting at index {i}: {e}")
            continue

    # Step 6: Finalize the Milvus collection.
    print("All batches inserted. Flushing data to make it searchable...")
    # `flush()` ensures that all data inserted into Milvus is sealed in segments and becomes searchable.
    collection.flush()
    print("Data flushed.")
    
    # Step 7: Build the vector index for fast queries.
    create_vector_index(collection)
    
    # Step 8: Load the collection into memory. This is crucial for achieving low-latency searches.
    print("Loading collection into memory for querying...")
    collection.load()
    
    print(f"Process complete. Final entity count in collection: {collection.num_entities}")

if __name__ == "__main__":
    main()