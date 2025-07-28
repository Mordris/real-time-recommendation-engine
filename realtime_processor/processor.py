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

# Add the project root to the Python path to allow importing the central 'config' module.
sys.path.append('..')
import config

# --- External Service Connection Helper ---
# It's a best practice to encapsulate external connections in a class.
# Bytewax will create one instance of this class for each worker process,
# ensuring that connections are established once per worker, not once per item.
class ExternalConnections:
    """Manages persistent connections to Milvus and Redis for a Bytewax worker."""
    def __init__(self):
        print("--- Initializing new external service connections ---")
        # Milvus Connection
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        self.milvus_collection = Collection(config.MILVUS_COLLECTION_NAME)
        
        # Redis Connection
        self.redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
        
    def get_item_vector(self, item_id: str):
        """Fetches a single item's vector from Milvus by its ID."""
        res = self.milvus_collection.query(expr=f"item_id == '{item_id}'", output_fields=["item_embedding"])
        return np.array(res[0]["item_embedding"]) if res else None

    def get_recommendations(self, vector, limit=10):
        """Performs a vector similarity search in Milvus to find the top N similar items."""
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
        """Writes a list of recommendation IDs to the Redis cache for a given user."""
        redis_key = f"{config.RECOMMENDATION_KEY_PREFIX}{user_id}"
        # Store as a JSON-encoded string for simple and universal access.
        self.redis_client.set(redis_key, json.dumps(recommendations))

# --- Bytewax Stateful Logic ---
# Instantiate the connection helper. Bytewax's execution model ensures this
# object is created once per worker and is available to the mapper functions.
connections_helper = ExternalConnections()

def update_and_recommend(user_vector, interaction):
    """
    This is the core stateful mapping function. It's called for each interaction event for a given user.
    - user_vector: The current state for the user's key (their taste vector).
    - interaction: The new data for that key (the interaction dictionary).
    Returns a tuple: (new_state, output_value)
    """
    item_id = interaction['item_id']
    item_vector = connections_helper.get_item_vector(item_id)

    # If the interacted item doesn't exist in Milvus, we can't update the profile.
    if item_vector is None:
        # Return the old state unmodified and an output message indicating the error.
        return user_vector, {"user_id": interaction["user_id"], "recommendations": [], "error": f"Item {item_id} not found"}

    # If this is the user's first interaction, their state (user_vector) will be None.
    # We initialize their taste profile with this first item's vector.
    if user_vector is None:
        new_user_vector = item_vector
    else:
        # For existing users, update their taste vector with a weighted moving average.
        # This gives more weight to their historical profile while adapting to the new interaction.
        new_user_vector = (user_vector * 0.9) + (item_vector * 0.1)
    
    # After updating the vector, immediately generate new recommendations.
    recommendations = connections_helper.get_recommendations(new_user_vector)
    # A common-sense rule: don't recommend the item the user just interacted with.
    if item_id in recommendations:
        recommendations.remove(item_id)

    # Return the updated vector as the new state, and a dictionary with the recommendations as the output.
    return new_user_vector, {
        "user_id": interaction["user_id"],
        "recommendations": recommendations
    }

def write_to_redis(user_id__recs_dict):
    """
    This is a "side-effect" mapper. It takes the output from the stateful step and
    writes the recommendations to Redis.
    """
    user_id, recs_dict = user_id__recs_dict
    # Only write to Redis if there wasn't an error in the previous step.
    if "error" not in recs_dict:
        connections_helper.write_recommendations_to_redis(user_id, recs_dict["recommendations"])
    return user_id__recs_dict # Pass the data through unchanged for the next step (logging).

# --- Bytewax Dataflow Definition ---
flow = Dataflow("recommendation_processor")

# Step 1: Input from Kafka (Redpanda). `tail=True` makes this a long-running, "live" stream.
kafka_stream = op.input("kafka_in", flow, KafkaSource(
    brokers=[config.KAFKA_BOOTSTRAP_SERVERS],
    topics=[config.USER_INTERACTIONS_TOPIC],
    tail=True 
))

# Step 2: Deserialize the Kafka message.
def decode_message(msg: KafkaSourceMessage):
    """Extracts key and value, decodes them, and parses the JSON value."""
    return (msg.key.decode('utf-8'), json.loads(msg.value.decode('utf-8')))

# The stream is now composed of (user_id, interaction_dict) tuples.
keyed_stream = op.map("decode", kafka_stream, decode_message)

# Step 3: Apply the stateful logic.
# Bytewax automatically groups the stream by the key (user_id) and applies our mapper.
stateful_stream = op.stateful_map(
    "update_and_recommend",
    keyed_stream,
    builder=lambda: None, # For each new user, their initial state is `None`.
    mapper=update_and_recommend
)

# Step 4: Write the results to Redis.
redis_stream = op.map("write_to_redis", stateful_stream, write_to_redis)

# Step 5: Format the output for logging to the console.
def format_output(user_id__recs_dict):
    """Creates a human-readable string for logging the outcome of the processing."""
    user_id, recs_dict = user_id__recs_dict
    if "error" in recs_dict:
        return f"⚠️  User: {user_id}, Error: {recs_dict['error']}"
    return f"✅ User: {user_id}, Wrote {len(recs_dict['recommendations'])} recs to Redis."

formatted_stream = op.map("format_output", redis_stream, format_output)

# Step 6: Output the formatted string to standard out.
op.output("stdout", formatted_stream, StdOutSink())