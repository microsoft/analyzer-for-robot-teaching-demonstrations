from pathlib import Path
import shutil
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import zipfile
import os.path as osp
import fastapi
from fastapi import Form, UploadFile, File
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi_utils import save_upload_file_to_tmp
import os
from pybsc import save_json
from pydub import AudioSegment
import json
import azure.cognitiveservices.speech as speechsdk
from utils.parsehandler import ParseHandler
import re
import nltk
import tempfile

nltk.download('omw-1.4')


def recognize_from_file(filename="audio.wav"):
    speech_config = speechsdk.SpeechConfig(
        subscription="SUBSCRIPTION_KEY",
        region="REGION")
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(
            speech_recognition_result.no_match_details))
        return ''
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print(
            "Speech Recognition canceled: {}".format(
                cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(
                "Error details: {}".format(
                    cancellation_details.error_details))
        return ''


__version__ = '0.0.1'

app = fastapi.FastAPI()

SERVICE = {
    "name": "speech-and-text_analyzer",
    "version": __version__,
    "libraries": {
        "speech_segmentation": __version__
    },
}


@app.post("/audio_split_and_speech_recognition/")
def speech_recognition(upload_file: UploadFile = File(None),
                       upload_json: UploadFile = File(None)):

    audio_file_path = save_upload_file_to_tmp(upload_file)
    with tempfile.TemporaryDirectory() as output_dir:
        song = AudioSegment.from_wav(str(audio_file_path))
        json_file_path = save_upload_file_to_tmp(upload_json)
        print(song.duration_seconds)
        # open the json file
        with open(json_file_path) as json_file:
            data = json.load(json_file)
        time_maximum = data['time_maximum']  # in seconds
        # add 0 to head
        time_maximum = [0] + time_maximum
        print(time_maximum)
        segment_time_start = []
        segment_time_end = []
        for i in range(len(time_maximum) - 1):
            t1 = time_maximum[i]
            t2 = time_maximum[i + 1]
            t2 = max(t1, min(t2, song.duration_seconds))
            newAudio = song[t1 * 1000:t2 * 1000]
            print('wav(t1,t2):{:.2f},{:.2f}'.format(t1, t2))
            segment_time_start.append(t1)
            segment_time_end.append(t2)
            # Exports to a wav file in the current path.
            newAudio.export(
                str(output_dir) +
                '/{}.wav'.format(i),
                format="wav")
        print(audio_file_path)
        recognized_text = []

        out_filepaths = []
        for i in range(len(time_maximum) - 1):
            #print(str(output_dir) + '/{}.wav'.format(i))
            text = recognize_from_file(
                str(output_dir) + '/{}.wav'.format(i))
            out_filepaths.append(str(output_dir) + '/{}.wav'.format(i))
            print(text)
            print('------------')
            recognized_text.append(text)

        segment_info = {}
        # numpy to list
        segment_info['recognized_text'] = recognized_text
        segment_info['segment_time_start'] = segment_time_start
        segment_info['segment_time_end'] = segment_time_end
        info_json_path = osp.join(output_dir, 'transcript.json')
        save_json(segment_info, info_json_path)

        out_filepaths.append(info_json_path)
        zip_filename = str(Path(output_dir).with_suffix('.zip'))
        print(f'Saving to zip file {zip_filename}')
        zf = zipfile.ZipFile(zip_filename, "w")
        for fpath in out_filepaths:
            print(f'Saving {fpath} to zip file {zip_filename}')
            fdir, fname = os.path.split(fpath)
            zf.write(fpath, fname)
            Path(fpath).unlink()
        shutil.rmtree(output_dir)
        if os.path.exists(audio_file_path):
            Path(audio_file_path).unlink()
        if os.path.exists(json_file_path):
            Path(json_file_path).unlink()
        return FileResponse(
            zip_filename,
            media_type='application/x-zip-compressed',
            filename=f'transcript.zip')


parse_handler = ParseHandler()


def extract_words(sentence, parsehandler):
    parsehandler.update_text(sentence)
    verbs = parsehandler.get_verb_list()
    object_list_parsehandlerformat = []
    verb_list = [v['word_lemmatized'] for v in verbs]
    object_list = [[item['word_lemmatized'] for item in parsehandler.resolve_object_children(
        v, relation="obj")] for v in verbs]
    for v in verbs:
        object_list_parsehandlerformat.extend(
            parsehandler.resolve_object_children(
                v, relation="obj"))
    attribute_list = [[item['word_lemmatized'] for item in parsehandler.resolve_object_children(
        o, relation="amod")] for o in object_list_parsehandlerformat]
    object_list = sum(object_list, [])
    attribute_list = sum(attribute_list, [])
    return verb_list, object_list, attribute_list


@app.post("/text_based_taskrecognition/")
def text_based_task_recognition(upload_json: UploadFile = File(None)):
    json_file_path = save_upload_file_to_tmp(upload_json)
    # open the json file
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    recognized_text = data['recognized_text']

    # load verb_task_dict.json
    with open('./library/verb_task_dict.json') as json_file:
        verb_task_dict = json.load(json_file)
    task_list = []
    for item in recognized_text:
        if item == "":
            task_list.append('NOTTASK')
            continue
        proc_sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(item).lower())
        verb_list, object_list, _ = extract_words(proc_sentence, parse_handler)
        print(proc_sentence)
        print(verb_list)
        print(object_list)
        print('-----------------')

        # insert if key exist in verb_task_dict
        if len(verb_list) > 0 and verb_list[0] in verb_task_dict.keys(
        ):  # focus one verb in a sentence
            task_list.append(verb_task_dict[verb_list[0]])
        else:
            task_list.append('UNKNOWN')
    if os.path.exists(json_file_path):
        Path(json_file_path).unlink()
    print(task_list)
    data['recognized_tasks'] = task_list
    return JSONResponse(content=jsonable_encoder(data))


@app.post("/text_based_objectrecognition/")
def text_based_objectrecognition(upload_json: UploadFile = File(None)):
    json_file_path = save_upload_file_to_tmp(upload_json)
    # open the json file
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    recognized_text = data['recognized_text']

    target_object = []
    target_attribute = []
    for item in recognized_text:
        proc_sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(item).lower())
        verb_list, object_list, attribute_list = extract_words(
            proc_sentence, parse_handler)
        print(proc_sentence)
        print(verb_list)
        print(object_list)
        print(attribute_list)

        if len(verb_list) > 0:  # focus one verb in a sentence
            if len(object_list) > 0:
                target_object.append(object_list[0])
                if len(attribute_list) > 0:
                    target_attribute.append(attribute_list[0])
                else:
                    target_attribute.append('UNKNOWN')
            else:
                target_object.append('UNKNOWN')
                target_attribute.append('UNKNOWN')
        else:
            target_object.append('UNKNOWN')
            target_attribute.append('UNKNOWN')
        print('-----------------')
    print(target_object)
    print(target_attribute)
    if os.path.exists(json_file_path):
        Path(json_file_path).unlink()
    data['recognized_objects'] = target_object
    data['recognized_attributes'] = target_attribute
    return JSONResponse(content=jsonable_encoder(data))


@app.post("/text_based_taskrecognition_luisbased/")
def text_based_task_recognition_luis(upload_json: UploadFile = File(None)):
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.language.conversations import ConversationAnalysisClient

    def analyze_input_with_score(query):
        client = ConversationAnalysisClient("ENDPOINT_URL", AzureKeyCredential("KEY"))
        with client:
            #query = "Send an email to Carol about the tomorrow's demo"
            result = client.analyze_conversation(
                task={
                    "kind": "Conversation",
                    "analysisInput": {
                        "conversationItem": {
                            "participantId": "1",
                            "id": "1",
                            "modality": "text",
                            "language": "en",
                            "text": query
                        },
                        "isLoggingEnabled": False
                    },
                    "parameters": {
                        "projectName": "LfO_task",
                        "deploymentName": "model2",
                        "verbose": True
                    }
                }
            )

        return result['result']['prediction']['topIntent'], result['result']['prediction']['intents'][0]['confidenceScore']
    json_file_path = save_upload_file_to_tmp(upload_json)
    # open the json file
    print('debugggg')
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    recognized_text = data['recognized_text']

    task_list = []
    for item in recognized_text:
        if item == "":
            task_list.append('NOTTASK')
            continue
        #import pdb;pdb.set_trace()
        response, score = analyze_input_with_score(item)
        task_tmp = response.upper()
        if task_tmp == 'NONE':
            task_tmp = 'UNKNOWN'
        task_list.append(task_tmp)
    if os.path.exists(json_file_path):
        Path(json_file_path).unlink()
    print(recognized_text)
    print(task_list)
    data['recognized_tasks'] = task_list
    return JSONResponse(content=jsonable_encoder(data))
