# Use official Python image
FROM python:3.11-slim

# Install system dependencies (needed for pdf2image -> poppler)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (Render uses PORT env variable)
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
