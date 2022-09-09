FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

LABEL maintainer="shinnosuke.matsuo@human.ait.kyushu-u.ac.jp"

# Timezone setting
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# Install something
RUN apt-get update && apt-get install -y --no-install-recommends bash curl fish git nano sudo

# RUN rm /usr/bin/python3
# RUN rm /usr/bin/python3.8

# OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev
RUN apt-get update && apt-get install -y python3-openslide

# Install Python
ENV PYTHON_VERSION 3
RUN apt-get update && apt-get install -y --no-install-recommends python${PYTHON_VERSION}

# Install pip
RUN curl --silent https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python${PYTHON_VERSION} get-pip.py
RUN rm get-pip.py
RUN pip install --upgrade pip

# Install Python library
COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install hydra-core --upgrade
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113