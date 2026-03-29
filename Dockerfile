# Use Python 3.10 slim for lightweight image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or individual files
COPY . .

# Install Python packages
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    pandas \
    numpy \
    scikit-learn \
    pvlib \
    fastapi \
    uvicorn \
    requests \
    streamlit \
    plotly

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command (will be overridden in docker-compose for each service)
CMD ["python3", "main.py"]
