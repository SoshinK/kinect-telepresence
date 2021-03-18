#!/usr/bin/env bash

source source.sh

VOLUMES="-v /home/kvsoshin:/home/kvsoshin -v /media/kvsoshin/Transcend:/media/kvsoshin/Transcend -v /tmp/.X11-unix:/tmp/.X11-unix"

sudo xhost +local:root

# ensure nvidia is your default runtime
docker run -ti --gpus all --privileged -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 $PARAMS $VOLUMES $NAME_03 $@
