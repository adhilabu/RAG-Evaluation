#!/bin/bash

# Stop all running services

echo "🛑 Stopping Document Processing System..."

# Kill uvicorn processes
pkill -f "uvicorn app.main:app" && echo "✅ Stopped FastAPI backend"

# Kill streamlit processes
pkill -f "streamlit run" && echo "✅ Stopped Streamlit frontend"

echo "✅ All services stopped"
