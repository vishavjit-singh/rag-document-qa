# Q&A API image. The SYNTHETIC corpus is generated and indexed at build time, and the abstention threshold is
# calibrated on the dev split, so the image answers questions as soon as it starts.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src RAGQA_HOME=/app
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ragqa.yaml ./
COPY src ./src
RUN python -m ragqa generate && python -m ragqa ingest && python -m ragqa eval --no-sweep

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["python", "-m", "ragqa", "serve", "--host", "0.0.0.0", "--port", "8000"]
