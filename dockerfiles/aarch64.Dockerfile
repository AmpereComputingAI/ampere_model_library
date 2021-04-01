FROM ubuntu:20.04

# set timezone
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt-get update

# apt-get installation
RUN apt-get install -y \
	python3-pip \
	wget \
	unzip \
	cmake

# pip packages
RUN pip3 install \
	numpy

# OpenCV installation
RUN cd / && wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.13.zip && unzip /opencv.zip
RUN cd / && wget https://github.com/opencv/opencv_contrib/archive/3.4.13.zip && unzip /3.4.13.zip
RUN cd /opencv-3.4.13 && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-3.4.13/modules/ ..
RUN cd /opencv-3.4.13/build && make -j && make install -j
RUN rm -R opencv-3.4.13 opencv.zip opencv_contrib-3.4.13 3.4.13.zip
