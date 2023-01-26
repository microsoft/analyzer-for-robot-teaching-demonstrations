#!/bin/bash

name="debug"
docker stop $name;
docker rm $name;
docker run --rm \
       --network=host \
       --privileged \
       --name="$name" \
       --volume="/dev:/dev" \
       --volume=$(pwd)/../src:/src \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       -it daemon \
       /bin/sh
