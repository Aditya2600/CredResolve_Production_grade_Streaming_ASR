FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
ENV NVIDIA_LIB=/opt/conda/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cuda_nvrtc/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/nvjitlink/lib:${LD_LIBRARY_PATH}
WORKDIR /srv

# git  — only for the torch.hub silero-vad fallback path. The preferred silero-vad
#        PyPI package (pinned in worker/requirements.txt) ships bundled weights and
#        does not need git or network access at runtime.
# Rust/Cargo — needed at pip-install time to compile deepfilterlib (the Rust extension
#              bundled inside the deepfilternet package). The toolchain is removed after
#              pip install so it does not inflate the final image at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl build-essential pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt /srv/requirements.txt
RUN python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
# Cargo is BUILD-TIME ONLY. deepfilterlib is the Rust extension inside the
# deepfilternet wheel that pip compiles from source on this base image; once
# the wheel is installed into site-packages we no longer need rustc/cargo,
# so the toolchain is removed in the same RUN layer to keep it out of the
# final image. Verified end-to-end by scripts/verify_worker_image.sh.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --no-modify-path && \
    PATH="/root/.cargo/bin:${PATH}" pip install -r /srv/requirements.txt && \
    rm -rf /root/.cargo /root/.rustup

COPY worker/app /srv/app
EXPOSE 9000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","9000","--workers","1"]
