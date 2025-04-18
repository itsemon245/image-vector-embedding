FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY app/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# Default fallback value - will be overridden by .env or docker-compose settings
ENV APP_ENV=production
ENV PORT=8787
ENV WORKERS=4
ENV MODEL_NAME=openai/clip-vit-base-patch32
ENV DEVICE=cpu

# Files from app are already copied as volume in docker-compose.yml


CMD ["./start.sh"]
