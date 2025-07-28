# /realtime_processor/processor.py
# The Bytewax dataflow for real-time processing of user interactions.

import json
import sys

from bytewax.dataflow import Dataflow
from bytewax import operators as op
from bytewax.connectors.kafka import KafkaSource, KafkaSourceMessage # Import the message type
from bytewax.connectors.stdio import StdOutSink

# Add the project root to the Python path to allow importing 'config'
sys.path.append('..')
import config

# --- Bytewax Dataflow Definition ---

flow = Dataflow("recommendation_processor")

# Step 1: Input from Kafka (Redpanda)
kafka_stream = op.input(
    "kafka_in", 
    flow, 
    KafkaSource(
        brokers=[config.KAFKA_BOOTSTRAP_SERVERS],
        topics=[config.USER_INTERACTIONS_TOPIC],
        tail=False 
    )
)

# Step 2: Decode the message value from bytes to a string
# *** THE FIX IS HERE: The input is a KafkaSourceMessage object. ***
# We access its `.value` attribute to get the message payload.
def decode_message(kafka_message: KafkaSourceMessage):
    return kafka_message.value.decode("utf-8")

decoded_stream = op.map("decode", kafka_stream, decode_message)

# Step 3: Parse the string as JSON
json_stream = op.map("json_parse", decoded_stream, json.loads)

# Step 4: Format the output string
def format_output(interaction_dict):
    return f"✅ Received Interaction: {interaction_dict}"

formatted_stream = op.map("format_output", json_stream, format_output)

# Step 5: Output to standard out
op.output("stdout", formatted_stream, StdOutSink())