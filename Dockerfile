FROM python:3.11-slim

WORKDIR /app
COPY main.py requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

CMD ["python", "main.py"]
