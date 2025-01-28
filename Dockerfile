# Use a base Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH" && \
    rustup default stable

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
