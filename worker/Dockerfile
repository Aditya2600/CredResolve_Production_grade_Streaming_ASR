FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
ENV NVIDIA_LIB=/opt/conda/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cuda_nvrtc/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/nvjitlink/lib:${LD_LIBRARY_PATH}
WORKDIR /srv
COPY worker/requirements.txt /srv/requirements.txt
COPY worker/requirements-en.txt /srv/requirements-en.txt
ARG INSTALL_EN_ENGINE=false
RUN python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
RUN pip install -r /srv/requirements.txt
RUN if [ "$INSTALL_EN_ENGINE" = "true" ]; then pip install -r /srv/requirements-en.txt; fi
COPY worker/app /srv/app
EXPOSE 9000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","9000","--workers","1"]
