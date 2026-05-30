FROM python:3.11-slim



RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip


WORKDIR /test

COPY requirements.txt .
COPY ./utils/ ./utils/


RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir ultralytics


RUN pip install --no-cache-dir -r requirements.txt

COPY script.py .


CMD ["python", "script.py"]
