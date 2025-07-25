# apis/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# --- ADDED: Install curl and netcat for robust startup checks ---
RUN apt-get update && apt-get install -y curl netcat-openbsd --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .