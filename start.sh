#!/bin/bash

if [ "$APP_ENV" = "production" ]; then
    echo "🚀 Starting server in ${APP_ENV} mode..."
    echo "Using model: ${MODEL_NAME} on device: ${DEVICE}"
    gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b "0.0.0.0:${PORT}"
    
else
    echo "🔧 Starting server in ${APP_ENV} mode..."
    echo "Using model: ${MODEL_NAME} on device: ${DEVICE}"
    python main.py
fi