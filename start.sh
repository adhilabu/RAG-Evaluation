#!/bin/bash

# Document Processing System - Startup Script
# This script starts both the FastAPI backend and Streamlit frontend

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Document Processing System...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Run: python3 -m venv venv${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found. Copy .env.example and add your OpenAI API key${NC}"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if OpenAI API key is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo -e "${RED}⚠️  Warning: OpenAI API key not set in .env file${NC}"
    echo -e "${RED}   Add your API key before uploading documents${NC}"
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}🛑 Shutting down services...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT SIGTERM

# Start backend in background
echo -e "${GREEN}📡 Starting FastAPI Backend...${NC}"
# Set PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8085 &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${BLUE}⏳ Waiting for backend to start...${NC}"
sleep 3

# Check if backend is running
if ! curl -s http://localhost:8085/health > /dev/null; then
    echo -e "${RED}❌ Backend failed to start${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}✅ Backend running at http://localhost:8085${NC}"
echo -e "${GREEN}📚 API docs available at http://localhost:8085/docs${NC}"

# Start frontend in foreground
echo -e "${GREEN}🎨 Starting Streamlit Frontend...${NC}"
cd frontend
streamlit run streamlit_app.py

# This line is reached when Streamlit is closed
cleanup
