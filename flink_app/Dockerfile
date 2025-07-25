# flink_app/Dockerfile

FROM apache/flink:1.17.2
USER root

# MODIFIED: Added netcat-openbsd to the install list for the startup check
RUN apt-get update && apt-get install -y python3-pip wget curl jq netcat-openbsd --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN wget -P /opt/flink/lib/ https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/1.17.2/flink-sql-connector-kafka-1.17.2.jar
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --no-cache-dir apache-flink==1.17.2

# Copy requirements before install to leverage cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /opt/flink/usrlib && chown -R flink:flink /opt/flink/usrlib

USER flink