import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sqlalchemy import create_engine
from tqdm import tqdm

# --- 1. Configuration ---
# Milvus Configuration
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'products'
DIMENSION = 384  # Dimension of the sentence-transformer model we are using

# PostgreSQL Configuration
POSTGRES_USER = 'admin'
POSTGRES_PASSWORD = 'admin'
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = '5432'
POSTGRES_DB = 'nebula_db'
TABLE_NAME = 'product_metadata'

# Data and Model Configuration
DATASET_URL = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz'
MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for downloading
MAX_PRODUCTS = 10000 # Limit the number of products to process for this example


def download_and_process_data():
    """Downloads and processes the dataset."""
    print("Downloading and processing dataset...")
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    # Decompress and read line by line
    data = []
    with requests.get(DATASET_URL, stream=True) as r:
        r.raise_for_status()
        # Use gzip and io to handle decompression on the fly
        import gzip
        import io
        gz = gzip.GzipFile(fileobj=io.BytesIO(r.content))
        for line in tqdm(gz, desc="Processing JSON data"):
            try:
                # Use eval as a simple (but be cautious) way to parse the pseudo-json
                record = eval(line)
                data.append(record)
                if len(data) >= MAX_PRODUCTS:
                    break
            except (SyntaxError, NameError):
                # Skip lines that are not valid Python literals
                continue

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} products.")

    # Data Cleaning
    df = df[['asin', 'title', 'description']].dropna()
    df = df.drop_duplicates(subset=['asin'])
    df['description'] = df['description'].str.strip()
    df = df[df['description'].str.len() > 20] # Keep only descriptions with some substance
    df = df.head(MAX_PRODUCTS).reset_index(drop=True)
    print(f"Cleaned data, {len(df)} products remaining.")
    return df


def create_postgres_connection():
    """Creates a connection to the PostgreSQL database."""
    engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')
    return engine


def load_metadata_to_postgres(df, engine):
    """Loads product metadata into PostgreSQL."""
    print("Loading metadata to PostgreSQL...")
    # Keep only metadata columns
    metadata_df = df[['asin', 'title', 'description']]
    metadata_df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
    print(f"Successfully loaded {len(metadata_df)} records into PostgreSQL table '{TABLE_NAME}'.")


def create_milvus_collection():
    """Creates a new collection in Milvus if it doesn't exist."""
    print("Connecting to Milvus...")
    connections.connect('default', host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Dropping it.")
        utility.drop_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name='asin', dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields, description='Product Embeddings')
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    print("Creating index on 'embedding' field...")
    index_params = {
        'metric_type': 'L2',
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 128}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    collection.load()
    print("Milvus collection setup complete.")
    return collection


def generate_and_insert_embeddings(df, collection):
    """Generates embeddings and inserts them into Milvus."""
    print(f"Loading sentence transformer model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings and inserting into Milvus in batches...")
    batch_size = 500
    for i in tqdm(range(0, len(df), batch_size), desc="Embedding and Inserting"):
        batch_df = df.iloc[i:i+batch_size]
        # Combining title and description for a richer embedding
        text_to_embed = (batch_df['title'] + ". " + batch_df['description']).tolist()
        embeddings = model.encode(text_to_embed)

        # Prepare data for Milvus insertion
        data = [
            batch_df.index.tolist(),
            batch_df['asin'].tolist(),
            embeddings
        ]
        collection.insert(data)

    collection.flush()
    print(f"\nSuccessfully inserted {len(df)} embeddings into Milvus.")
    print(f"Total entities in collection: {collection.num_entities}")


if __name__ == "__main__":
    # 1. Get data
    product_df = download_and_process_data()

    # 2. Setup PostgreSQL
    pg_engine = create_postgres_connection()
    load_metadata_to_postgres(product_df, pg_engine)

    # 3. Setup Milvus
    milvus_collection = create_milvus_collection()

    # 4. Generate and insert embeddings
    generate_and_insert_embeddings(product_df, milvus_collection)

    print("\nOffline pipeline finished successfully!")