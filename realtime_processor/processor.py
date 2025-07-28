# /realtime_processor/processor.py
# The Bytewax dataflow for real-time processing of user interactions.

import json
import sys
import numpy as np
import redis

from bytewax.dataflow import Dataflow
from bytewax import operators as op
from bytewax.connectors.kafka import KafkaSource, KafkaSourceMessage
from bytewax.connectors.stdio import StdOutSink

from pymilvus import Collection, connections

sys.path.append('..')
import config

# --- External Service Connection Helper ---
class ExternalConnections:
    def __init__(self):
        print("--- Initializing new external service connections ---")
        # Milvus Connection
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        self.milvus_collection = Collection(config.MILVUS_COLLECTION_NAME)
        
        # Redis Connection
        self.redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
        
    def get_item_vector(self, item_id: str):
        res = self.milvus_collection.query(expr=f"item_id == '{item_id}'", output_fields=["item_embedding"])
        return np.array(res[0]["item_embedding"]) if res else None

    def get_recommendations(self, vector, limit=10):
        search_params = {"metric_type": "L2", "params": {"ef": 128}}
        results = self.milvus_collection.search(
            data=[vector.tolist()],
            anns_field="item_embedding",
            param=search_params,
            limit=limit,
            output_fields=["item_id"]
        )
        return [hit.entity.get('item_id') for hit in results[0]]

    def write_recommendations_to_redis(self, user_id, recommendations):
        redis_key = f"{config.RECOMMENDATION_KEY_PREFIX}{user_id}"
        # Store as a JSON string
        self.redis_client.set(redis_key, json.dumps(recommendations))

# --- Bytewax Stateful Logic ---
connections_helper = ExternalConnections()

def update_and_recommend(user_vector, interaction):
    item_id = interaction['item_id']
    item_vector = connections_helper.get_item_vector(item_id)

    if item_vector is None:
        return user_vector, {"user_id": interaction["user_id"], "recommendations": [], "error": f"Item {item_id} not found"}

    new_user_vector = item_vector if user_vector is None else (user_vector * 0.9) + (item_vector * 0.1)
    
    # After updating the vector, immediately generate new recommendations
    recommendations = connections_helper.get_recommendations(new_user_vector)
    # Exclude the item the user just interacted with from the recommendations
    if item_id in recommendations:
        recommendations.remove(item_id)

    return new_user_vector, {
        "user_id": interaction["user_id"],
        "recommendations": recommendations
    }

def write_to_redis(user_id__recs_dict):
    user_id, recs_dict = user_id__recs_dict
    if "error" not in recs_dict:
        connections_helper.write_recommendations_to_redis(user_id, recs_dict["recommendations"])
    return user_id__recs_dict # Pass through for logging

# --- Bytewax Dataflow Definition ---
flow = Dataflow("recommendation_processor")

kafka_stream = op.input("kafka_in", flow, KafkaSource(
    brokers=[config.KAFKA_BOOTSTRAP_SERVERS],
    topics=[config.USER_INTERACTIONS_TOPIC],
    tail=True 
))

def decode_message(msg: KafkaSourceMessage):
    return (msg.key.decode('utf-8'), json.loads(msg.value.decode('utf-8')))

keyed_stream = op.map("decode", kafka_stream, decode_message)

stateful_stream = op.stateful_map(
    "update_and_recommend",
    keyed_stream,
    builder=lambda: None,
    mapper=update_and_recommend
)

# This map step is a "side effect" that writes the results to Redis
redis_stream = op.map("write_to_redis", stateful_stream, write_to_redis)

def format_output(user_id__recs_dict):
    user_id, recs_dict = user_id__recs_dict
    if "error" in recs_dict:
        return f"⚠️  User: {user_id}, Error: {recs_dict['error']}"
    return f"✅ User: {user_id}, Wrote {len(recs_dict['recommendations'])} recs to Redis."

formatted_stream = op.map("format_output", redis_stream, format_output)

op.output("stdout", formatted_stream, StdOutSink())