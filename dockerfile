# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy entire project
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Run the app
CMD ["python3", "app.py"]