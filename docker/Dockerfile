FROM python:3.8.12-slim-buster

# bash utils
RUN apt-get update && apt-get install -y gnupg2
RUN apt-get update && apt-get install -y \
    build-essential \
    less byobu \
    ssh xvfb \
    emacs

RUN pip install --no-cache-dir --upgrade pip
RUN pip install matplotlib pygame scipy
RUN pip install pandas scikit-learn
RUN pip install nltk
RUN pip install jupyter
RUN pip install transformers

# downloads for nltk
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader wordnet

# Download Stanford NLP tools
RUN apt-get update && apt-get install -y wget zip unzip
ENV STANFORD_NLP_ROOT /stanford
RUN mkdir -p ${STANFORD_NLP_ROOT} \
    && cd ${STANFORD_NLP_ROOT} \
    && wget https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip \
    && wget https://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-english.jar

RUN cd ${STANFORD_NLP_ROOT} \
    && unzip stanford-parser-4.2.0.zip \ 
    && rm stanford-parser-4.2.0.zip
# downloads for web API
RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn

RUN pip install torch

RUN apt-get update && apt-get install -y default-jre

RUN pip install python-multipart
RUN pip install aiofiles
RUN pip install opencv-python
RUN pip install pybsc
RUN pip install sk-video
RUN pip install pydub
RUN pip install azure-cognitiveservices-speech
RUN pip install ffprobe

RUN pip install open3d
RUN pip install trimesh
RUN pip install azure-core
RUN pip install azure-ai-language-conversations