FROM python:3.11-slim

# Install ffmpeg for Whisper audio processing
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p data/index

# Expose Chainlit port
EXPOSE 7860

# Run Chainlit
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
