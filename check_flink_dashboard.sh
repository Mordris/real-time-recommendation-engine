#!/bin/bash

# check_flink_dashboard.sh
# Script to verify Flink dashboard accessibility

echo "=== Flink Dashboard Health Check ==="

# Check if Flink JobManager is running
echo "1. Checking JobManager container status..."
if docker ps | grep -q "flink-jobmanager"; then
    echo "✅ JobManager container is running"
else
    echo "❌ JobManager container is not running"
    exit 1
fi

# Check if port 8081 is accessible
echo "2. Checking port 8081 accessibility..."
if curl -s -f -o /dev/null http://localhost:8081; then
    echo "✅ Port 8081 is accessible"
else
    echo "❌ Port 8081 is not accessible"
    echo "   Checking if port is bound..."
    netstat -tlnp | grep :8081 || echo "   Port 8081 is not bound"
    exit 1
fi

# Check Flink REST API
echo "3. Checking Flink REST API..."
RESPONSE=$(curl -s http://localhost:8081/overview)
if echo "$RESPONSE" | grep -q "taskmanagers"; then
    echo "✅ Flink REST API is responding"
else
    echo "❌ Flink REST API is not responding properly"
    echo "   Response: $RESPONSE"
    exit 1
fi

# Check TaskManager registration
echo "4. Checking TaskManager registration..."
TM_COUNT=$(curl -s http://localhost:8081/taskmanagers | grep -o '"id":' | wc -l)
if [ "$TM_COUNT" -gt 0 ]; then
    echo "✅ $TM_COUNT TaskManager(s) registered"
else
    echo "⚠️  No TaskManagers registered yet"
fi

# Check for running jobs
echo "5. Checking for running jobs..."
JOBS=$(curl -s http://localhost:8081/jobs)
if echo "$JOBS" | grep -q '"jobs"'; then
    JOB_COUNT=$(echo "$JOBS" | grep -o '"id":' | wc -l)
    echo "✅ Found $JOB_COUNT job(s)"
    if [ "$JOB_COUNT" -gt 0 ]; then
        echo "$JOBS" | jq '.jobs[] | {id: .id, name: .name, state: .state}' 2>/dev/null || echo "   (jq not available for pretty printing)"
    fi
else
    echo "⚠️  No jobs found"
fi

echo ""
echo "=== Dashboard Access ==="
echo "🌐 Flink Dashboard: http://localhost:8081"
echo "📊 Overview: http://localhost:8081/#/overview"
echo "📋 Job List: http://localhost:8081/#/job-list"
echo ""
echo "=== Container Logs ==="
echo "To view JobManager logs: docker logs flink-jobmanager"
echo "To view TaskManager logs: docker logs flink-taskmanager"