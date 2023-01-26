from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import os.path as osp
import fastapi
import datetime
from fastapi import Form, UploadFile, File
from pathlib import Path
import tempfile
from tempfile import NamedTemporaryFile
from fastapi.responses import FileResponse
from pybsc import save_json
from fastapi_utils import save_upload_file_to_tmp
import os
import tempfile
import time

__version__ = '0.0.1'

app = fastapi.FastAPI()

SERVICE = {
    "name": "video_to_laban_converter",
    "version": __version__,
    "libraries": {
        "luminance_based_video_segmentation": __version__
    },
}


@app.get("/")
async def get_root():
    return {
        "service": SERVICE,
        "time": int(datetime.datetime.now().timestamp() * 1000),
    }


@app.get('/info', tags=['Utility'])
async def info():
    about = dict(
        version=__version__,
    )
    return about

@app.post("/luminance_based_video_segmentation/")
def convert_video(upload_file: UploadFile = File(None),
                  fs: int = Form(30),
                  return_full: bool = Form(False)):
    t1 = time.time()
    video_file_path = save_upload_file_to_tmp(upload_file)
    video_name = Path(upload_file.filename).stem

    with tempfile.TemporaryDirectory() as output_dir:
        cap = cv2.VideoCapture(str(video_file_path))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        lum_level = np.zeros(length)
        prev_frame = None
        _, frame = cap.read()
        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if return_full:
            writer = cv2.VideoWriter(
                str(output_dir) + '/luminance.mp4', fourcc, int(fs), (2 * w, h), 0)
        while frame is not None:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            lum = yuv[:, :, 0]
            if prev_frame is None:
                prev_frame = 0 * lum
                print('None frame')
            # calculate the difference in uint16
            lum_uint16 = lum.astype(np.uint16)
            prev_lum_uint16 = prev_frame.astype(np.uint16)
            # abs difference

            dif_uint16 = np.abs(cv2.blur(lum_uint16, (50, 50)) -
                                cv2.blur(prev_lum_uint16, (50, 50)))
            dif = dif_uint16.astype(np.uint8)
            dif = cv2.blur(dif, (50, 50))
            prev_frame = lum
            if return_full:
                # concatenate the difference
                show_frame = np.concatenate((lum, dif), axis=1)
                writer.write(show_frame)
            lum_level[int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1] = np.mean(dif)
            _, frame = cap.read()
        cap.release()
        if return_full:
            writer.release()
        print(lum_level)
        # post processing
        lum_level[0] = 0
        lum_level[lum_level > np.mean(
            lum_level) + 3 * np.std(lum_level)] = np.nan
        for i in range(len(lum_level)):
            if np.isnan(lum_level[i]):
                lum_level[i] = lum_level[i - 1]

        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(
                order, normal_cutoff, btype='low', analog=False)
            return b, a

        from scipy import signal
        b, a = butter_lowpass(0.3, fs)
        lum_level_lpf = signal.filtfilt(b, a, lum_level, padlen=int(fs))

        # plot the luminance level
        import matplotlib.pyplot as plt
        t = np.arange(0,len(lum_level)/fs,1.0/fs)
        plt.plot(t, lum_level)
        plt.plot(t, lum_level_lpf)
        plt.ylabel('Delta of luminance level')
        plt.xlabel('Time')
        plt.legend(['raw', 'filtered'])

        # save the plot as a png file
        plt.savefig(str(output_dir) + '/lum_level.png')
        plt.clf()

        # find segment times
        diff_lum_level = np.diff(lum_level_lpf)
        # append none to head
        diff_lum_level_1 = np.insert(diff_lum_level, 0, 0)
        # append none to tail
        diff_lum_level_2 = np.append(diff_lum_level, 0)
        sel_minumum = np.where(
            np.logical_and(
                diff_lum_level_1 < 0,
                diff_lum_level_2 > 0))
        sel_minumum = sel_minumum[0]
        # add 0 to head
        sel_minumum = np.insert(sel_minumum, 0, 0)
        # add length to tail
        sel_minumum = np.append(sel_minumum, length)

        sel_maximum = np.where(
            np.logical_and(
                diff_lum_level_1 > 0,
                diff_lum_level_2 < 0))
        sel_maximum = sel_maximum[0]

        out_filepaths = []
        if return_full:
            video_save_name = str(output_dir) + '/luminance.mp4'
            out_filepaths.append(video_save_name)

        image_save_name = str(output_dir) + '/lum_level.png'
        out_filepaths.append(image_save_name)

        org_video_path = osp.join(output_dir,
                                  f'{video_name}{video_file_path.suffix}')

        segment_info = {}
        # numpy to list
        sel_time_minimum = sel_minumum / float(fs)
        sel_time_maximum = sel_maximum / float(fs)
        segment_info['frame_minimum'] = sel_minumum.tolist()
        segment_info['time_minimum'] = sel_time_minimum.tolist()
        segment_info['frame_maximum'] = sel_maximum.tolist()
        segment_info['time_maximum'] = sel_time_maximum.tolist()
        info_json_path = osp.join(output_dir, 'segment.json')
        save_json(segment_info, info_json_path)
        out_filepaths.append(info_json_path)
        print(segment_info)

        if return_full:
            # split the video based on sel
            cap = cv2.VideoCapture(str(video_file_path))
            print(str(video_file_path))
            ret, frame = cap.read()
            h, w, _ = frame.shape
            parts = []
            for i in range(len(sel_minumum) - 1):
                parts.append((sel_minumum[i], sel_minumum[i + 1]))
            writer = cv2.VideoWriter(
                str(output_dir) + '/segment_result.mp4', fourcc, int(fs), (w, h))
            out_filepaths.append(str(output_dir) + '/segment_result.mp4')
            f = 0
            while ret:
                f += 1
                currentseg = None
                for i, part in enumerate(parts):
                    start, end = part
                    if start <= f <= end:
                        currentseg = i
                # draw currentseg info to the frame
                if currentseg is not None:
                    cv2.putText(
                        frame,
                        f"Segment index: {currentseg}",
                        (10,
                         60),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        4,
                        color=(
                            255,
                            255,
                            255),
                        thickness=4)
                writer.write(frame)
                ret, frame = cap.read()
            writer.release()
            cap.release()
        zip_subdir = f"{video_name}"
        zip_filename = str(Path(output_dir).with_suffix('.zip'))
        print(f'Saving to zip file {zip_filename}')
        zf = zipfile.ZipFile(zip_filename, "w")
        for fpath in out_filepaths:
            print(f'Saving {fpath} to zip file {zip_filename}')
            fdir, fname = os.path.split(fpath)
            zip_path = os.path.join(zip_subdir, fname)
            zf.write(fpath, zip_path)
            Path(fpath).unlink()

        zf.close()
        shutil.move(video_file_path, org_video_path)
        # delete the tmp folders
        shutil.rmtree(output_dir)
        t2 = time.time()
        elapsed_time = t2-t1
        print(f"Elapsed time:{elapsed_time}")
        return FileResponse(
            zip_filename,
            media_type='application/x-zip-compressed',
            filename=f'{video_name}.zip')
