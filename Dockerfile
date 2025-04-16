FROM python:3.10-slim

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
