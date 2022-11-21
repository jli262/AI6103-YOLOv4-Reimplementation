from typing import List

import cv2 as cv
import numpy as np
from imutils.video import FPS, WebcamVideoStream
from net.yolo import Yolo


def image_detect(model_path: str, image_path: str, classes: List[str],
                 anchors=None, conf_thresh=0.6, use_gpu=True, show_conf=True):
    model = Yolo(len(classes), anchors, conf_thresh)
    if use_gpu:
        model = model.cuda()

    model.load(model_path)
    model.eval()

    return model.detect(image_path, classes,  show_conf)


def camera_detect(model_path: str, classes: List[str], anchors: list = None,
                  camera_src=0, conf_thresh=0.6, use_gpu=True):
    model = Yolo(len(classes), anchors, conf_thresh)
    if use_gpu:
        model = model.cuda()

    model.load(model_path)
    model.eval()

    fps = FPS().start()

    print('Detecting Object, Press Q to Quit')

    stream = WebcamVideoStream(src=camera_src).start()
    while True:
        image = stream.read()
        image = np.array(model.detect(image, classes, use_gpu))
        fps.update()

        cv.imshow('camera detection', image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print(f'Check End, FPS : {fps.fps()} FPS')
