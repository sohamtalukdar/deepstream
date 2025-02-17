#!/bin/bash

xhost +local:
sudo docker run --gpus all -it --rm --network=host --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    -v /home/soham/deepstream:/workspace -w /workspace \
    nvcr.io/nvidia/deepstream:7.1-gc-triton-devel