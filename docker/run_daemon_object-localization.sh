#!/bin/bash

name="daemon-object-localization"
docker stop $name;
docker rm $name;
docker run --rm \
       --network=host \
       --privileged \
       --name="$name" \
       --volume="/dev:/dev" \
       --volume=$(pwd)/../src:/src \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       -it arrey:daemon \
       /bin/sh -c 'cd /src && uvicorn daemon_object-localization:app --reload --host 0.0.0.0 --port 8086'
       
