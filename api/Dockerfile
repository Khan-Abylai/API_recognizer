FROM nvcr.io/nvidia/tensorrt:20.09-py3

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LANGUAGE=C.UTF-8

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install \
    tzdata sudo \
    locales-all \
    build-essential \
    gcc \
    g++ \
    gdb \
    clang \
    make \
    ninja-build \
    cmake \
    autoconf \
    automake \
    dos2unix \
    rsync \
    tar \
    python \
    python3-dev \
    python3-pip \
    ffmpeg libsm6 libxext6 \
  && apt-get clean

RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py && python3 get-pip.py --force-reinstall
RUN pip3 install tqdm llvmlite numba scipy opencv-python uvicorn fastapi python-multipart \
    Flask matplotlib numpy python-telegram-bot requests scikit-image scipy pycocotools nltk gunicorn shortuuid

ADD src/ /opt/api/
WORKDIR /opt/api/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["sh", "entrypoint.sh"]
