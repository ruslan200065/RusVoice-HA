FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 ORT_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libstdc++6 espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/rusvoice
COPY requirements.txt /opt/rusvoice/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app/ /opt/rusvoice/app/
EXPOSE 10200 10300 8099
CMD ["python", "-m", "app.server"]
