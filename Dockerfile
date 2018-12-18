FROM continuumio/anaconda3:5.2.0
ENV SERVICE_NAME=digit-recognizer

MAINTAINER Kugatov Maxim <maximkugatov@gmail.com>

#fastai
COPY /fastai "/usr/src/${SERVICE_NAME}/fastai"
RUN conda env update -f "/usr/src/${SERVICE_NAME}/fastai/environment-cpu.yml"

ENV PATH /opt/conda/envs/fastai-cpu/bin:$PATH
RUN echo "conda activate fastai-cpu" >> ~/.bashrc

#dependencies
RUN python -m pip install grpcio && python -m pip install grpcio-tools && python -m pip install matplotlib \
     && python -m pip install pyqt5

#app
COPY /models "/usr/src/${SERVICE_NAME}/models"
COPY /tmp "/usr/src/${SERVICE_NAME}/tmp"
COPY /recognizer "/usr/src/${SERVICE_NAME}/recognizer"
COPY /server "/usr/src/${SERVICE_NAME}/server"
COPY run.py "/usr/src/${SERVICE_NAME}"

WORKDIR /usr/src/$SERVICE_NAME

#RUN conda activate fastai-cpu

EXPOSE 50051

ENTRYPOINT ["python", "run.py"]
