FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python data/generate_synthetic.py && python -m src.ingest
ENV OLLAMA_URL=http://ollama:11434 LLM_MODEL=qwen2.5:7b-instruct
CMD ["python", "-m", "src.app", "chat"]
