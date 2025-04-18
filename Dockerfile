FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY app/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the start script from root directory
COPY start.sh .
RUN chmod +x start.sh

# Default fallback value - will be overridden by .env or docker-compose settings
ENV APP_ENV=production
ENV PORT=8787

CMD ["./start.sh"]
