# Nebula - Real-Time Recommendation Engine

## Project Vision

**Nebula** is a state-of-the-art, end-to-end recommendation system that updates a user's recommendations in real-time as they interact with a product catalog. The system leverages stateful stream processing to dynamically update a user's "taste profile" (represented as a vector) and uses a high-performance vector database for instantaneous similarity search to find relevant items.

This project is designed to be a portfolio centerpiece, showcasing modern data engineering and MLOps practices.

---

## System Architecture

The system consists of two main pipelines: an **Offline Pipeline** for initial data preparation and an **Online Pipeline** for real-time processing and serving.

### Architectural Diagram

![Nebula Architecture](https://i.imgur.com/your-diagram-image-url.png)
_(You can create and upload a diagram later to a service like Imgur and replace this link)_

### Data Flow

1.  **Offline Pipeline (Batch Processing)**:

    - **Input**: A dataset of products with text descriptions.
    - **Process**: A Python script using `sentence-transformers` converts product descriptions into dense vector embeddings.
    - **Output**: The embeddings are bulk-loaded into a **Milvus** vector database. Product metadata is stored in **PostgreSQL**.

2.  **Online Pipeline (Real-Time Streaming)**:

    - **(A) User Interaction**: A user interacts with an item. The interaction event (`user_id`, `item_id`) is sent to a **FastAPI** endpoint.
    - **(B) Event Ingestion**: The API produces the event to a **Redpanda** (Kafka) topic.
    - **(C) Stateful Processing**: An **Apache Flink** job consumes the events, keyed by `user_id`.
      - **State Management**: Flink maintains a "taste vector" for each user in its managed state.
      - **Dynamic Update**: Flink fetches the item's vector from Milvus and updates the user's taste vector (e.g., using a moving average).
    - **(D) Candidate Generation**: The Flink job uses the newly updated user vector to query Milvus for the Top-K most similar items.
    - **(E) Caching Results**: The list of recommended item IDs is published to a **Redis** cache.

3.  **Serving Layer**:
    - **Recommendation API**: A second **FastAPI** endpoint (`/recommendations/{user_id}`) serves recommendations by fetching the pre-computed list directly from Redis for ultra-low latency.
    - **Real-Time UI**: A **Streamlit** dashboard simulates user interactions and visualizes how recommendations change in real-time.

---

## Technology Stack

| Category              | Technology                                  | Purpose                                                     |
| --------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| **Stream Processing** | Apache Flink (PyFlink)                      | Stateful computation and real-time ML updates.              |
| **Message Broker**    | Redpanda                                    | High-performance, Kafka-compatible message bus.             |
| **Vector Database**   | Milvus                                      | Storing and indexing item embeddings for similarity search. |
| **Caching & Serving** | Redis                                       | Low-latency storage for final recommendation lists.         |
| **APIs & Backend**    | Python, FastAPI                             | API for event ingestion and serving recommendations.        |
| **ML & Data Prep**    | Hugging Face (SentenceTransformers), Pandas | Generating embeddings and data preparation.                 |
| **Orchestration**     | Docker & Docker Compose                     | Containerizing and managing all services.                   |
| **Real-Time UI**      | Streamlit                                   | Interactive dashboard for visualization and demo.           |

---

## How to Run

1.  **Start Infrastructure**:

    ```bash
    docker compose up -d
    ```

2.  **Run Offline Pipeline**:
    _Navigate to the `offline_pipeline` directory and run the script to populate the databases._

    ```bash
    cd offline_pipeline
    pip install -r requirements.txt
    python batch_embedder.py
    ```

3.  **Start the System Components**:
    _(Instructions to be added for running the API, Flink job, and UI)_
