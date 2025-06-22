# Use official Python image as base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy app code
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn transformers torch

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
