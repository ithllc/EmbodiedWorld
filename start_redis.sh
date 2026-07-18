#!/usr/bin/env bash
# Start the workspace-local Redis daemon. Safe to run multiple times.
set -euo pipefail
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
if redis-cli -p 6379 ping >/dev/null 2>&1; then
    echo "Redis already running on localhost:6379"
    exit 0
fi
redis-server "$WORKSPACE/redis.conf" --daemonize yes
sleep 1
redis-cli -p 6379 ping
echo "Redis started. Config: $WORKSPACE/redis.conf  Log: $WORKSPACE/logs/redis.log"
