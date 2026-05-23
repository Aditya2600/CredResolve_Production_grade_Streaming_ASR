ARG TRITON_SERVER_IMAGE=nvcr.io/nvidia/tritonserver:24.09-py3
FROM ${TRITON_SERVER_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY triton/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt

COPY triton/model_repository /models

CMD ["tritonserver", "--model-repository=/models", "--model-control-mode=none", "--strict-readiness=true"]
