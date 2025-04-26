FROM python:3.10-slim

ENV POETRY_VERSION=1.8.3
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CXXFLAGS='-std=c++11'
ENV PYTHONPATH=/app/src
ENV PARLANT_HOME=/app/data

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Copy project files
COPY . .

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry lock --no-update \
    && poetry install --no-interaction --no-ansi

# Expose the port your app runs on
EXPOSE 8800

# Command to run the server
CMD ["poetry", "run", "python", "src/parlant/bin/server.py", "run", "--log-level", "debug"]
