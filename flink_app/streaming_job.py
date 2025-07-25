# flink_app/streaming_job.py (FINAL CORRECTED VERSION)

import json
import logging
import numpy as np

# Import connections and utility from pymilvus
from pymilvus import Collection, connections, utility
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.serialization import SimpleStringSchema

# --- Configuration ---
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'products'
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'user_interactions'
KAFKA_GROUP_ID = 'flink-recommendation-consumer'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserStateUpdater(KeyedProcessFunction):
    def __init__(self):
        self.user_taste_vector_state = None
        self.task_name = None

    def open(self, runtime_context: RuntimeContext):
        """
        Initialize state handle. No remote connections are made here.
        """
        self.task_name = runtime_context.get_task_name_with_subtasks()
        descriptor = ValueStateDescriptor("user_taste_vector", Types.PICKLED_BYTE_ARRAY())
        self.user_taste_vector_state = runtime_context.get_state(descriptor)
        logger.info(f"[{self.task_name}] UserStateUpdater opened successfully.")

    def process_element(self, value, ctx: KeyedProcessFunction.Context):
        """
        Handle one event. Connection is managed entirely within this method.
        """
        alias = f"milvus_conn_{self.task_name}"
        try:
            # 1. Connect to Milvus for each element. This is the most robust approach.
            # CORRECTED LINE: Use connections.has_connection
            if not connections.has_connection(alias):
                connections.connect(alias, host=MILVUS_HOST, port=MILVUS_PORT)
            
            milvus_collection = Collection(COLLECTION_NAME, using=alias)
            if not milvus_collection.is_loaded:
                milvus_collection.load()

            event = json.loads(value)
            user_id = event['user_id']
            item_id = int(event['item_id'])
            logger.info(f"[{self.task_name}] Processing interaction for user '{user_id}', item '{item_id}'.")

            # 2. Perform the logic
            item_results = milvus_collection.query(expr=f"id == {item_id}", output_fields=["embedding"], timeout=10)

            if not item_results:
                logger.warning(f"[{user_id}] Item {item_id} not found. Skipping.")
                return

            item_vector = np.array(item_results[0]['embedding'], dtype=np.float32)
            current_vector = self.user_taste_vector_state.value()

            if current_vector is None:
                new_vector = item_vector
                logger.info(f"[{user_id}] First interaction. Initializing taste vector.")
            else:
                current_vector = np.array(current_vector, dtype=np.float32)
                new_vector = (current_vector * 0.8) + (item_vector * 0.2)
                logger.info(f"[{user_id}] Updated taste vector.")

            self.user_taste_vector_state.update(new_vector)
            logger.info(f"[{user_id}] New taste vector (first 5 dims): {new_vector[:5]}")

        except Exception as e:
            logger.error(f"[{self.task_name}] Error processing interaction {value}: {e}", exc_info=True)
        finally:
            # 3. Always disconnect to prevent connection leaks.
            # CORRECTED LINE: Use connections.has_connection
            if connections.has_connection(alias):
                connections.disconnect(alias)

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.enable_checkpointing(5000)
    kafka_props = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'latest'
    }
    kafka_source = FlinkKafkaConsumer(
        topics=KAFKA_TOPIC,
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_props
    )
    data_stream = env.add_source(kafka_source)
    data_stream \
        .key_by(lambda msg: json.loads(msg)['user_id']) \
        .process(UserStateUpdater()) \
        .name("UserStateUpdater")

    logger.info("Submitting Flink job: Nebula_Stateful_Vector_Updater")
    env.execute("Nebula_Stateful_Vector_Updater")

if __name__ == '__main__':
    main()