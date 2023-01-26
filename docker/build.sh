#!/bin/bash

docker build --network=host -t daemon -f ./Dockerfile .