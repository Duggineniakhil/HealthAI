# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV/Matplotlib
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Assume models are mounted or copied. 
# Better practice is to keep models outside and mount them via volume.

EXPOSE 8000

CMD ["python", "main.py"]
