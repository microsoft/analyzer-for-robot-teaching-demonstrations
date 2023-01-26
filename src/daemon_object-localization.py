from time import time, time_ns
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import fastapi
from fastapi import Form, UploadFile, File
from tempfile import NamedTemporaryFile
from fastapi_utils import save_upload_file_to_tmp
import cv2
import asyncio

__version__ = '0.0.1'

app = fastapi.FastAPI()

SERVICE = {
    "name": "hand_localizer",
    "version": __version__,
    "libraries": {
        "hand_localizer": __version__
    },
}


def detect_objects(buf):
    url = 'ENDPOINT_URL'

    headers = {'content-type': 'application/octet-stream',
               'Prediction-Key': 'PREDICTION_KEY'}
    threshold = 0.3
    import requests
    response = requests.post(url, data=buf.tobytes(), headers=headers)
    response.raise_for_status()
    analysis = response.json()
    predicts = []
    for item in analysis["predictions"]:
        if item["probability"] > threshold:
            predicts.append({"name": item["tagName"],
                             "probability": item["probability"],
                             "location": item["boundingBox"]})
    return predicts


async def run_detection(loop, buf_list):
    sem = asyncio.Semaphore(8)  # you may need to adjust this number depending on the allowed number of API requests per second

    async def run_request(buf):
        async with sem:
            return await loop.run_in_executor(None, detect_objects, buf)
    predicts_list = [run_request(buf) for buf in buf_list]
    return await asyncio.gather(*predicts_list)


async def run_classify(loop, buf_list):
    sem = asyncio.Semaphore(5)  # you may need to adjust this number depending on the allowed number of API requests per second

    async def run_request(buf):
        async with sem:
            return await loop.run_in_executor(None, classify_objects, buf)
    predicts_list = [run_request(buf) for buf in buf_list]
    return await asyncio.gather(*predicts_list)


def get_object_loc(predicts, object_name):
    loc = None
    prob = 0.3
    for item in predicts:
        if item["name"].lower() == object_name.lower():
            if item["probability"] > prob:
                prob = item["probability"]
                loc = item["location"]
    return loc


def get_object_list(predicts):
    ret = []
    prob = 0.3
    for item in predicts:
        if item["probability"] > prob:
            ret.append(item["name"].lower())
    return ret


@app.post("/object_localization_image/")
def hand_localization_image(upload_file: UploadFile = File(None),
                            object_name: str = Form('none')):
    print(f'input object name:{object_name}')
    print('come detection...')
    image_file_path = save_upload_file_to_tmp(upload_file)
    frame = cv2.imread(str(image_file_path))
    _, buf = cv2.imencode('.jpg', frame)
    predicts = detect_objects(buf)
    info_frame = {}
    loc_object = {}
    loc = get_object_loc(predicts, object_name)
    list_objects = get_object_list(predicts)
    if loc is not None:
        x_min = int(loc["left"] * frame.shape[1])
        x_max = int(loc["width"] * frame.shape[1]) + x_min
        y_min = int(loc["top"] * frame.shape[0])
        y_max = int(loc["height"] * frame.shape[0]) + y_min
        loc_object['left'] = x_min
        loc_object['top'] = y_min
        loc_object['right'] = x_max
        loc_object['bottom'] = y_max
    info_frame['location'] = loc_object
    info_frame['found_objects'] = list_objects
    print('end detection...')
    return JSONResponse(content=jsonable_encoder(info_frame))
