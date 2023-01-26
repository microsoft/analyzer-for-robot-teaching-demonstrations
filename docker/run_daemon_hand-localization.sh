#!/bin/bash

name="daemon-hand-localization"
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
       /bin/sh -c 'cd /src && uvicorn daemon_hand-localization:app --reload --host 0.0.0.0 --port 8083'
       
