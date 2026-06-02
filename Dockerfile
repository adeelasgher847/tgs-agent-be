# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for psycopg2
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . /app

# PORT defaults to 8000; override via environment variable at runtime.
ENV PORT=8000
EXPOSE ${PORT}

# Use shell form so $PORT is expanded at container start time.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
