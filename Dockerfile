FROM tensorflow/tensorflow:latest-gpu
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /opt/project
