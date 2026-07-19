#!/bin/bash

# Move to the volume
cd /workspace

echo "Setting up repository..."
if [ ! -d "lingbot-world-v2" ]; then
    git clone https://github.com/robbyant/lingbot-world-v2.git
fi

cd lingbot-world-v2

echo "Installing requirements..."
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install fastapi uvicorn pydantic "huggingface_hub[cli]"

echo "Downloading the 14B model (this might take a while)..."
huggingface-cli download robbyant/lingbot-world-v2-14b-causal-fast --local-dir ./lingbot-world-v2-14b-causal-fast

echo "Starting FastAPI server on Port 8888..."
nohup python3 /workspace/runpod_api_server.py > /workspace/api_server.log 2>&1 &
echo "Server started."
