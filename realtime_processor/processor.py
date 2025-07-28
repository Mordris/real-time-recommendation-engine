# /realtime_processor/processor.py
# The Bytewax dataflow for real-time processing of user interactions.

import json
import sys
import numpy as np

from bytewax.dataflow import Dataflow
from bytewax import operators as op
from bytewax.connectors.kafka import KafkaSource, KafkaSourceMessage
from bytewax.connectors.stdio import StdOutSink

from pymilvus import Collection, connections

# Add the project root to the Python path to allow importing 'config'
sys.path.append('..')
import config

# --- Milvus Connection Helper ---
class MilvusConnection:
    def __init__(self):
        print("--- MilvusConnection: Initializing new connection ---")
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        self.collection = Collection(config.MILVUS_COLLECTION_NAME)
        
    def get_item_vector(self, item_id: str):
        res = self.collection.query(
            expr=f"item_id == '{item_id}'",
            output_fields=["item_embedding"]
        )
        if not res:
            return None
        return np.array(res[0]["item_embedding"])

# --- Bytewax Stateful Logic ---
milvus_con = MilvusConnection()

def update_user_vector(user_vector, interaction):
    item_id = interaction['item_id']
    item_vector = milvus_con.get_item_vector(item_id)

    if item_vector is None:
        print(f"Warning: Item ID '{item_id}' not found in Milvus. Skipping update.")
        return user_vector, f"Item {item_id} not found"

    if user_vector is None:
        new_user_vector = item_vector
    else:
        new_user_vector = (user_vector * 0.9) + (item_vector * 0.1)
    
    return new_user_vector, {
        "user_id": interaction["user_id"],
        "updated_vector": new_user_vector.tolist()
    }

# --- Bytewax Dataflow Definition ---
flow = Dataflow("recommendation_processor")

kafka_stream = op.input(
    "kafka_in", 
    flow, 
    KafkaSource(
        brokers=[config.KAFKA_BOOTSTRAP_SERVERS],
        topics=[config.USER_INTERACTIONS_TOPIC],
        # *** THE FIX IS HERE: `tail=True` makes the process long-running. ***
        tail=True 
    )
)

def decode_message(msg: KafkaSourceMessage):
    return (msg.key.decode('utf-8'), json.loads(msg.value.decode('utf-8')))

keyed_stream = op.map("decode", kafka_stream, decode_message)

stateful_stream = op.stateful_map(
    "update_taste_vector",
    keyed_stream,
    builder=lambda: None,
    mapper=update_user_vector
)

def format_output(user_id__update_info):
    user_id, update_info = user_id__update_info
    vector_preview = update_info['updated_vector'][:4]
    return f"✅ User: {user_id}, Updated Vector (preview): {vector_preview}..."

formatted_stream = op.map("format", stateful_stream, format_output)

op.output("stdout", formatted_stream, StdOutSink())