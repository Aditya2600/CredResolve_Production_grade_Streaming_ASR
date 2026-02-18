#!/bin/bash

PROM_URL=${PROM_URL:-"http://localhost:9090"}
QUERY=$1

if [ -z "$QUERY" ]; then
    echo "Usage: $0 '<prometheus_query>'"
    exit 1
fi

curl -sG --data-urlencode "query=$QUERY" "$PROM_URL/api/v1/query" | jq .data.result
