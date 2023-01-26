# Explanations of the scripts in this directory
## build.sh
This script builds the docker image. The image is named `daemon` by default. You can change the name by editing the script.
## run_daemon_extract-roi-position.sh
Give the set of rgb image, depth image and region of interest, this script runs the daemon for extracting the 3D position of the region of interest.

## run_daemon_extract-hand-localization.sh
Given an rgb image, this script runs the daemon for extracting the 2D region of interest of the hand. This code uses azure congitive service for hand detection, so you need to set the key and endpoint in the script.

## run_daemon_extract-object-localization.sh
Given an rgb image, this script runs the daemon for extracting the 2D region of interest of the object. This code uses azure congitive service for object detection, so you need to set the key and endpoint in the script.

## run_daemon_spech recognition.sh
Given an audio file, this script runs the daemon for extracting the text from the audio file. This script also extact the target object name and recognizes tasks to be performed by robots. This code uses azure congitive service for speech recognition, so you need to set the key and endpoint in the script.

## run_daemon_video-segmentation.sh
Given an video file, this script runs the daemon for splitting the video into multiple segments based on the intensity of the hand movement. Please see our [paper](https://arxiv.org/abs/2212.10787) for details.