# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install backend dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install frontend dependencies
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy frontend code
COPY frontend/ .

# Expose the port
EXPOSE 8080

# Command to run the backend server
CMD ["python", "app.py"]
