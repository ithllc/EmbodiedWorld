#!/usr/bin/env bash
# Cleanly stop the workspace-local Redis daemon (does not touch system services).
set -euo pipefail
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$WORKSPACE/redis_data/redis.pid"
if [[ -f "$PIDFILE" ]]; then
    PID="$(cat "$PIDFILE")"
    if ps -p "$PID" >/dev/null 2>&1; then
        echo "Stopping Redis (pid $PID) ..."
        redis-cli -p 6379 shutdown nosave 2>/dev/null || kill "$PID" || true
        sleep 1
    fi
    rm -f "$PIDFILE"
fi
if redis-cli -p 6379 ping >/dev/null 2>&1; then
    echo "WARN: Redis still responding on 6379 (may be a different instance)."
else
    echo "Redis stopped."
fi
