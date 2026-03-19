FROM python:3.11-bullseye

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN mkdir -p artifacts
COPY artifacts/model.pkl artifacts/model.pkl

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
