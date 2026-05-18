FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

# ✅ Needed to compile webrtcvad (C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
  && rm -rf /var/lib/apt/lists/*

COPY gateway/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY gateway/app /srv/app
COPY itn_service/__init__.py /srv/itn_service/__init__.py
COPY itn_service/service /srv/itn_service/service
RUN bash /srv/itn_service/service/build_protos.sh

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
