# Nebula - Real-Time Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![Apache Flink](https://img.shields.io/badge/Apache%20Flink-1.17-orange?logo=apache) ![Kafka](https://img.shields.io/badge/Redpanda-Kafka%20API-black?logo=kafka) ![FastAPI](https://img.shields.io/badge/FastAPI-blueviolet?logo=fastapi) ![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker) ![Milvus](https://img.shields.io/badge/Milvus-2.3-green?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJtMTY3LjQ5IDEyOC4wNjUtNzkuNDEgNDUuODQ0djE2LjA0M2w5NS40NTItNTUuMTEzVjE5LjczMmwtOTUuNDUyIDU1LjExMlY5MC44ODlsNzkuNDEtNDUuODQ0djgzeiIgZmlsbD0iI0ZGRiIvPjwvc3ZnPg==)

**Nebula** is a state-of-the-art, end-to-end recommendation system that dynamically updates a user's recommendations in real-time. As a user interacts with products, their "taste profile"—represented as a vector—is instantly updated, providing a new set of relevant recommendations on their next request.

This project is a portfolio centerpiece designed to showcase modern data engineering, MLOps, and real-time machine learning practices using a powerful, containerized stack.

---

## Key Features

- **Real-Time User Profile Updates**: User taste vectors are updated in milliseconds using a stateful Flink job.
- **High-Performance Vector Search**: Leverages Milvus for sub-second similarity search on millions of item embeddings.
- **Low-Latency Serving**: Recommendations are pre-calculated and cached in Redis, allowing the API to respond in under 50ms.
- **Scalable & Resilient Infrastructure**: Fully containerized with Docker Compose, featuring a high-throughput Redpanda message broker.
- **End-to-End Data Flow**: From batch data embedding to real-time event ingestion and final recommendation serving.

---

## System Architecture

The system is composed of two primary workflows: an **Offline Pipeline** for bootstrapping the system with data and an **Online Pipeline** for real-time event processing and recommendation serving.

### How It Works

#### 1. Offline Data Bootstrapping

Before the system can serve recommendations, it needs a catalog of items with corresponding vector embeddings.

- **Data Ingestion**: A Python script downloads and processes a product dataset.
- **Embedding Generation**: Using a `sentence-transformers` model, it converts product titles and descriptions into 384-dimensional vector embeddings.
- **Data Loading**: The script bulk-loads the product vectors and metadata into their respective databases:
  - **Milvus**: Stores the vector embeddings for fast similarity search.
  - **PostgreSQL**: Stores the product metadata (like title, description, and ASIN).

#### 2. Real-Time Interaction Loop

This is the core real-time component of the system.

1.  **Event Ingestion**: A user interaction (e.g., a 'click') is sent to a **FastAPI** endpoint. The API validates the event and produces it as a JSON message to a **Redpanda** (Kafka) topic.
2.  **Stateful Stream Processing**: An **Apache Flink** job consumes from the Redpanda topic.
    - For each event, Flink retrieves the user's current "taste vector" from its internal, fault-tolerant state.
    - It fetches the vector for the interacted item from **Milvus**.
    - It updates the user's taste vector using a weighted moving average and saves the new vector back to its state.
3.  **Real-Time Recommendation Generation**:
    - The Flink job immediately uses this new taste vector to query Milvus for the top-K most similar items.
    - This list of recommended item IDs is then pushed to a **Redis** cache, keyed by `user_id`.
4.  **Serving Recommendations**:
    - When the user requests their recommendations, a separate FastAPI endpoint (`/recommendations/{user_id}`) is called.
    - The API fetches the pre-computed list of item IDs directly from **Redis**, ensuring an ultra-low latency response.
    - It then enriches these IDs with metadata from **PostgreSQL** to provide the final recommendation details to the user.

---

## Technology Stack

| Category              | Technology                          | Purpose                                                             |
| --------------------- | ----------------------------------- | ------------------------------------------------------------------- |
| **Stream Processing** | Apache Flink (PyFlink)              | Powers the real-time, stateful vector calculations.                 |
| **Message Broker**    | Redpanda                            | Provides a high-performance, Kafka-compatible message bus.          |
| **Vector Database**   | Milvus                              | Stores and indexes embeddings for lightning-fast similarity search. |
| **Metadata Storage**  | PostgreSQL                          | Stores structured product metadata (titles, descriptions, etc.).    |
| **Caching & Serving** | Redis                               | Caches final recommendation lists for low-latency API reads.        |
| **APIs & Backend**    | Python & FastAPI                    | Manages event ingestion and serves recommendations.                 |
| **ML & Data Prep**    | Hugging Face (SentenceTransformers) | Generates high-quality vector embeddings from text.                 |
| **Orchestration**     | Docker & Docker Compose             | Containerizes and manages all microservices.                        |
| **Real-Time UI**      | Streamlit                           | Provides an interactive dashboard for demos and visualization.      |

---

## Getting Started

### Prerequisites

- **Docker** and **Docker Compose** installed.
- A machine with at least 8GB of RAM is recommended.

### 1. Configuration

This project uses an `.env` file for configuration. Create one from the example:

```bash
cp .env.example .env
```

_(No changes are needed to the default `.env` file to run locally.)_

### 2. Launch the System

All services, including the databases, Redpanda, and the Flink cluster, are managed by Docker Compose.

```bash
# Build and start all services in the background
docker compose up -d --build
```

This will take a few minutes the first time as it downloads and builds the necessary images. You can check the status of all services with `docker compose ps`.

### 3. Run the Offline Pipeline

Before the system can process real-time events, you must populate Milvus and PostgreSQL with the product data.

_**Note:** This script will first download a ~178 MB dataset file. This is a one-time operation._

```bash
# Run the batch embedding script from the project root
python -m offline_pipeline.batch_embedder
```

### 4. Interact with the System

The system is now running and ready to process events.

**A. Send a User Interaction Event**
Use `curl` or any API client to send a sample interaction. This simulates a user clicking on an item.

```bash
curl -X POST "http://localhost:8000/interaction" \
-H "Content-Type: application/json" \
-d '{"user_id": "user-42", "item_id": 150, "event_type": "click"}'
```

**B. Monitor the Flink Dashboard**
Open the Flink Dashboard in your browser at **[http://localhost:8081](http://localhost:8081)**. Navigate to "Running Jobs" and click on the `Nebula_Stateful_Vector_Updater` job. You will see the "Records Received" and "Records Sent" counters increase, confirming that your event was processed.

---

## Project Status: Phase 2 Complete

- [x] **Phase 1: Infrastructure & Offline Pipeline**: All services are containerized. The offline pipeline successfully populates Milvus and PostgreSQL.
- [x] **Phase 2: Real-Time Event Processing**: The API ingests events to Redpanda. The Flink job is stable, consumes events, updates user state, and queries Milvus.
- [ ] **Phase 3: Closing the Loop & UI**:
  - Implement the Flink sink to write recommendations to Redis.
  - Build the `/recommendations/{user_id}` API endpoint.
  - Develop the Streamlit UI.
