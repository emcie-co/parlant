FROM python:3.12-slim

ENV POETRY_VERSION=1.8.3
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CXXFLAGS='-std=c++11'
ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*
RUN pip install poetry==$POETRY_VERSION

COPY pyproject.toml poetry.lock /app/
COPY src /app/src
WORKDIR /app
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main

# Expose the port your app runs on
EXPOSE 8000

CMD ["poetry", "run", "emcie-server", "run"]
