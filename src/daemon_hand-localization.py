from operator import index
from pathlib import Path
import shutil
from time import time, time_ns
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import zipfile
import cv2
import os.path as osp
import fastapi
from fastapi import Form, UploadFile, File
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi.responses import FileResponse
from pybsc import save_json
from fastapi_utils import save_upload_file_to_tmp
import cv2
import os
import json
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


def classify_objects(buf):
    url = 'ENDPOINT_URL'
    headers = {'content-type': 'application/octet-stream',
               'Prediction-Key': 'PREDICTION_KEY'}
    result = "Unknown"
    import requests
    if buf is not None:
        response = requests.post(url, data=buf.tobytes(), headers=headers)
        response.raise_for_status()
        analysis = response.json()
        threshold = 0
        for item in analysis["predictions"]:
            if item["probability"] > threshold:
                result = item["tagName"]
                threshold = item["probability"]
        return (result, threshold)
    else:
        return (result, None)


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


def get_left_loc(predicts):
    left_loc = None
    prob = 0.3
    for item in predicts:
        if item["name"] == "LeftHand":
            if item["probability"] > prob:
                prob = item["probability"]
                left_loc = item["location"]
    return left_loc


def get_right_loc(predicts):
    right_loc = None
    prob = 0.3
    for item in predicts:
        if item["name"] == "RightHand":
            if item["probability"] > prob:
                prob = item["probability"]
                right_loc = item["location"]
    return right_loc


@app.post("/hand_localization/")
def hand_localization(upload_file: UploadFile = File(None),
                      fs: int = Form(30),
                      upload_json: UploadFile = File(None)):
    output_dir = '/tmp/hand_localization'
    output_dir_images = '/tmp/hand_localization/images'
    output_dir_images_debug = '/tmp/hand_localization/images_debug'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)
    if not os.path.exists(output_dir_images_debug):
        os.makedirs(output_dir_images_debug)

    video_file_path = save_upload_file_to_tmp(upload_file)
    video_name = Path(upload_file.filename).stem
    json_file_path = save_upload_file_to_tmp(upload_json)
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    time_focus_sec = data['time_focus']

    # convert second to frame index
    time_focus = [int(fs * t) for t in time_focus_sec]
    cap = cv2.VideoCapture(str(video_file_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # remove invalid time_focus
    time_focus = [t for t in time_focus if t < length]
    print('time focusing...')
    print(time_focus)

    _, frame = cap.read()

    buf_list = []
    frame_list = []

    for frame in time_focus:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame = cap.read()
        _, buf = cv2.imencode('.jpg', frame)
        buf_list.append(buf)
        frame_list.append(frame)
    cap.release()

    predicts_allframes = [detect_objects(buf) for buf in buf_list]
    info_hands = []
    file_paths = []

    for i, (predicts, frame) in enumerate(zip(predicts_allframes, frame_list)):
        frame_draw = frame.copy()
        info_hands_frame = {}
        loc_info_left = {}
        loc = get_left_loc(predicts)
        if loc is not None:
            x_min = int(loc["left"] * frame.shape[1])
            x_max = int(loc["width"] * frame.shape[1]) + x_min
            y_min = int(loc["top"] * frame.shape[0])
            y_max = int(loc["height"] * frame.shape[0]) + y_min
            frame_crop_left = frame[y_min:y_max, x_min:x_max]
            cv2.imwrite(f'{output_dir_images}/left_{i}.jpg', frame_crop_left)
            frame_draw = cv2.rectangle(
                frame_draw, pt1=(
                    x_min, y_min), pt2=(
                    x_max, y_max), color=(
                    255, 0, 0), thickness=5)
            loc_info_left['left'] = x_min
            loc_info_left['top'] = y_min
            loc_info_left['right'] = x_max
            loc_info_left['bottom'] = y_max
            loc_info_left['file_name'] = f'images/left_{i}.jpg'
            file_paths.append(f'{output_dir_images}/left_{i}.jpg')
        loc_info_right = {}
        loc = get_right_loc(predicts)
        if loc is not None:
            x_min = int(loc["left"] * frame.shape[1])
            x_max = int(loc["width"] * frame.shape[1]) + x_min
            y_min = int(loc["top"] * frame.shape[0])
            y_max = int(loc["height"] * frame.shape[0]) + y_min
            frame_crop_right = frame[y_min:y_max, x_min:x_max]
            cv2.imwrite(f'{output_dir_images}/right_{i}.jpg', frame_crop_right)
            frame_draw = cv2.rectangle(
                frame_draw, pt1=(
                    x_min, y_min), pt2=(
                    x_max, y_max), color=(
                    0, 255, 0), thickness=5)
            loc_info_right['left'] = x_min
            loc_info_right['top'] = y_min
            loc_info_right['right'] = x_max
            loc_info_right['bottom'] = y_max
            loc_info_right['file_name'] = f'images/right_{i}.jpg'
            file_paths.append(f'{output_dir_images}/right_{i}.jpg')

        info_hands_frame['location_lefthand'] = loc_info_left
        info_hands_frame['location_righthand'] = loc_info_right
        info_hands_frame['time_focus_sec'] = time_focus[i] / fs
        info_hands.append(info_hands_frame)
        if frame_draw is not None:
            cv2.imwrite(f'{output_dir_images_debug}/{i}.jpg', frame_draw)
        else:
            cv2.imwrite(f'{output_dir_images_debug}/{i}.jpg', frame)
        file_paths.append(f'{output_dir_images_debug}/{i}.jpg')
    info_json_path = osp.join(output_dir, 'hand_detection.json')
    save_json(info_hands, info_json_path)
    file_paths.append(info_json_path)

    zip_subdir = f"{video_name}"
    zip_filename = str(Path(output_dir).with_suffix('.zip'))
    print(f'Saving to zip file {zip_filename}')
    zf = zipfile.ZipFile(zip_filename, "w")
    for fpath in file_paths:
        print(f'Saving {fpath} to zip file {zip_filename}')
        fdir, fname = os.path.split(fpath)
        if 'debug' in fpath:
            zip_path = os.path.join(zip_subdir, 'images_debug', fname)
        elif fname.endswith('.jpg'):
            zip_path = os.path.join(zip_subdir, 'images', fname)
        elif fname.endswith('.json'):
            zip_path = os.path.join(zip_subdir, fname)
        zf.write(fpath, zip_path)
        Path(fpath).unlink()

    zf.close()
    # delete the tmp folder
    shutil.rmtree(output_dir)
    return FileResponse(
        zip_filename,
        media_type='application/x-zip-compressed',
        filename=f'{video_name}.zip')


@app.post("/hand_localization_image/")
def hand_localization_image(upload_file: UploadFile = File(None)):
    image_file_path = save_upload_file_to_tmp(upload_file)
    frame = cv2.imread(str(image_file_path))
    _, buf = cv2.imencode('.jpg', frame)
    predicts = detect_objects(buf)
    #import pdb; pdb.set_trace()
    info_hands_frame = {}
    loc_info_left = {}
    loc = get_left_loc(predicts)
    print("localization come")
    if loc is not None:
        x_min = int(loc["left"] * frame.shape[1])
        x_max = int(loc["width"] * frame.shape[1]) + x_min
        y_min = int(loc["top"] * frame.shape[0])
        y_max = int(loc["height"] * frame.shape[0]) + y_min
        loc_info_left['left'] = x_min
        loc_info_left['top'] = y_min
        loc_info_left['right'] = x_max
        loc_info_left['bottom'] = y_max
    loc_info_right = {}
    loc = get_right_loc(predicts)
    if loc is not None:
        x_min = int(loc["left"] * frame.shape[1])
        x_max = int(loc["width"] * frame.shape[1]) + x_min
        y_min = int(loc["top"] * frame.shape[0])
        y_max = int(loc["height"] * frame.shape[0]) + y_min
        loc_info_right['left'] = x_min
        loc_info_right['top'] = y_min
        loc_info_right['right'] = x_max
        loc_info_right['bottom'] = y_max

    info_hands_frame['location_lefthand'] = loc_info_left
    info_hands_frame['location_righthand'] = loc_info_right
    # remove image_file_path
    Path(image_file_path).unlink()
    print("localization finish")
    return JSONResponse(content=jsonable_encoder(info_hands_frame))


@app.post("/hand_localization_with_grasp/")
def hand_localization(upload_file: UploadFile = File(None),
                      fs: int = Form(30),
                      upload_json: UploadFile = File(None)):
    output_dir = '/tmp/hand_localization_with_grasp'
    output_dir_images = '/tmp/hand_localization_with_grasp/images'
    output_dir_images_debug = '/tmp/hand_localization_with_grasp/images_debug'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)
    if not os.path.exists(output_dir_images_debug):
        os.makedirs(output_dir_images_debug)

    video_file_path = save_upload_file_to_tmp(upload_file)
    video_name = Path(upload_file.filename).stem
    json_file_path = save_upload_file_to_tmp(upload_json)
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    time_focus_sec = data['time_focus']

    # convert second to frame index
    time_focus = [int(fs * t) for t in time_focus_sec]
    # read video
    cap = cv2.VideoCapture(str(video_file_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # remove invalid time_focus
    time_focus = [t for t in time_focus if t < length]
    print('time focusing...')
    print(time_focus)

    _, frame = cap.read()

    buf_list = []
    frame_list = []

    for frame in time_focus:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame = cap.read()
        _, buf = cv2.imencode('.jpg', frame)
        buf_list.append(buf)
        frame_list.append(frame)
    cap.release()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    predicts_allframes = loop.run_until_complete(run_detection(loop, buf_list))
    info_hands = []
    file_paths = []
    buf_list_left = []
    buf_list_right = []
    index_find_left = []
    index_find_right = []
    for i, (predicts, frame) in enumerate(zip(predicts_allframes, frame_list)):
        frame_draw = frame.copy()
        info_hands_frame = {}
        loc_info_left = {}
        grasp_left = {}
        loc = get_left_loc(predicts)
        if loc is not None:
            index_find_left.append(i)
            x_min = int(loc["left"] * frame.shape[1])
            x_max = int(loc["width"] * frame.shape[1]) + x_min
            y_min = int(loc["top"] * frame.shape[0])
            y_max = int(loc["height"] * frame.shape[0]) + y_min
            frame_crop_left = frame[y_min:y_max, x_min:x_max]
            cv2.imwrite(f'{output_dir_images}/left_{i}.jpg', frame_crop_left)
            frame_draw = cv2.rectangle(
                frame_draw, pt1=(
                    x_min, y_min), pt2=(
                    x_max, y_max), color=(
                    255, 0, 0), thickness=5)
            loc_info_left['left'] = x_min
            loc_info_left['top'] = y_min
            loc_info_left['right'] = x_max
            loc_info_left['bottom'] = y_max
            loc_info_left['file_name'] = f'images/left_{i}.jpg'
            file_paths.append(f'{output_dir_images}/left_{i}.jpg')

            _, buf = cv2.imencode('.jpg', frame_crop_left)
            buf_list_left.append(buf)
        loc_info_right = {}
        loc = get_right_loc(predicts)
        if loc is not None:
            index_find_right.append(i)
            x_min = int(loc["left"] * frame.shape[1])
            x_max = int(loc["width"] * frame.shape[1]) + x_min
            y_min = int(loc["top"] * frame.shape[0])
            y_max = int(loc["height"] * frame.shape[0]) + y_min
            frame_crop_right = frame[y_min:y_max, x_min:x_max]
            cv2.imwrite(f'{output_dir_images}/right_{i}.jpg', frame_crop_right)
            frame_draw = cv2.rectangle(
                frame_draw, pt1=(
                    x_min, y_min), pt2=(
                    x_max, y_max), color=(
                    0, 255, 0), thickness=5)
            loc_info_right['left'] = x_min
            loc_info_right['top'] = y_min
            loc_info_right['right'] = x_max
            loc_info_right['bottom'] = y_max
            loc_info_right['file_name'] = f'images/right_{i}.jpg'
            file_paths.append(f'{output_dir_images}/right_{i}.jpg')

            _, buf = cv2.imencode('.jpg', frame_crop_right)
            buf_list_right.append(buf)
            # file_paths.append(f'{output_dir_images_debug}/right_{i}.jpg')
        info_hands_frame['location_lefthand'] = loc_info_left
        info_hands_frame['location_righthand'] = loc_info_right
        info_hands_frame['time_focus_sec'] = time_focus[i] / fs
        info_hands.append(info_hands_frame)
        if frame_draw is not None:
            cv2.imwrite(f'{output_dir_images_debug}/{i}.jpg', frame_draw)
        else:
            cv2.imwrite(f'{output_dir_images_debug}/{i}.jpg', frame)
        file_paths.append(f'{output_dir_images_debug}/{i}.jpg')
    info_json_path = osp.join(output_dir, 'hand_detection.json')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    predicts_left = loop.run_until_complete(run_classify(loop, buf_list_left))
    for i, item in enumerate(predicts_left):
        info_hands[index_find_left[i]]['predict_lefthand'] = item
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    predicts_right = loop.run_until_complete(run_classify(loop, buf_list_right))
    for i, item in enumerate(predicts_right):
        info_hands[index_find_right[i]]['predict_righthand'] = item
    save_json(info_hands, info_json_path)
    file_paths.append(info_json_path)

    zip_subdir = f"{video_name}"
    zip_filename = str(Path(output_dir).with_suffix('.zip'))
    print(f'Saving to zip file {zip_filename}')
    zf = zipfile.ZipFile(zip_filename, "w")
    for fpath in file_paths:
        print(f'Saving {fpath} to zip file {zip_filename}')
        fdir, fname = os.path.split(fpath)
        if 'debug' in fpath:
            zip_path = os.path.join(zip_subdir, 'images_debug', fname)
        elif fname.endswith('.jpg'):
            zip_path = os.path.join(zip_subdir, 'images', fname)
        elif fname.endswith('.json'):
            zip_path = os.path.join(zip_subdir, fname)
        zf.write(fpath, zip_path)
        Path(fpath).unlink()

    zf.close()
    # delete the tmp folder
    shutil.rmtree(output_dir)
    return FileResponse(
        zip_filename,
        media_type='application/x-zip-compressed',
        filename=f'{video_name}.zip')


@app.post("/classify_grasp_release_image/")
def hand_localization_with_grasp_image(upload_file: UploadFile = File(None)):
    image_file_path = save_upload_file_to_tmp(upload_file)
    frame = cv2.imread(str(image_file_path))
    _, buf = cv2.imencode('.jpg', frame)
    predicts = classify_objects(buf)
    result = {}
    result['predict'] = predicts[0]
    result['confidence'] = str(predicts[1])
    return JSONResponse(content=jsonable_encoder(result))
