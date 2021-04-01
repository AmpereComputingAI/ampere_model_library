FROM ubuntu:20.04

# set timezone
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt-get update

# pip installation
RUN apt-get install -y python3-pip

# pip packages
RUN pip3 install \
	numpy \
	opencv-python \
