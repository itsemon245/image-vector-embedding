#!/bin/bash

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RESET='\033[0m'

echo -e "${YELLOW}󰥔 Starting server in ${APP_ENV} mode...${RESET}"
echo -e "${GREEN}󰥔 It could take a while to start...${RESET}"
if [ "$APP_ENV" = "production" ]; then
    gunicorn main:app -w $WORKERS -k uvicorn.workers.UvicornWorker -b "0.0.0.0:${PORT}"
    
else
    python main.py
fi