FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get --yes update && apt-get --yes install git

RUN pip install git+git://github.com/marco-willi/camera-trap-classifier

ENTRYPOINT ["bash"]
