#!/usr/bin/env python3
# source https://google.github.io/mediapipe/solutions/hands.html#model_complexity

import cv2
import mediapipe
import time
import multiprocessing
import os
import pandas as pd
import pickle
import argparse
import numpy as np
from imutils.video import WebcamVideoStream

FRAME_BUFFER_SIZE = 4


mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_hands = mediapipe.solutions.hands


def worker(image_queue, result_queue):
    with mp_hands.Hands(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            image = image_queue.get()
            start_time = time.time()
            if image is None:
                break
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result_queue.put(results.multi_hand_landmarks)


def start_workers(num_workers):
    pool = multiprocessing.Pool(processes=num_workers)
    m = multiprocessing.Manager()
    image_queue = m.Queue()
    result_queue = m.Queue()
    workers = [
        pool.apply_async(worker, (image_queue, result_queue))
        for _ in range(num_workers)
    ]
    return image_queue, result_queue, workers


#



image_queue, result_queue, workers = start_workers(num_workers=8)

num_buffer_frames = 0

cap = WebcamVideoStream(src=-1)
cap.start()

points_list = None
key_code = None
current_input_char_key_code = None
max_index = None
last_added_training_samples = []
avg_debug_times = []
train_data = []
num_train_steps = 0
recognized_str = ""
all_data_dataframe = pd.DataFrame()


with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:
    time_last_update = time.time()
    start_time = time.time()
    num_frames = 0
    while True:
        if True:
            read_start = time.time()
            start_time_debug = time.time()
            image = cap.read()
            num_frames += 1
            fps = num_frames / (time.time() - start_time)
            out_str = f"fps: {fps:.0f} num_frames: {num_frames}"

            print(out_str)
            iter_time = time.time() - time_last_update
            time_last_update = time.time()
            fps = 1 / iter_time
            if False:
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

            get_image_start = time.time()
            image.flags.writeable = False
            if False:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image).multi_hand_landmarks
            else:
                time1 = time.time()
                image_queue.put(image)
                if num_buffer_frames < FRAME_BUFFER_SIZE:
                    num_buffer_frames += 1
                    continue
                else:
                    time2 = time.time()
                    get_image_start = time.time()
                    results = result_queue.get()

        if num_frames % 10 == 0:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results:
                for hand_landmarks in results:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    l1 = hand_landmarks.landmark[0]
                    points_list = [
                        (datapoint.x - l1.x, datapoint.y - l1.y, datapoint.z)
                        for datapoint in hand_landmarks.landmark
                    ]
            if True:
                cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
                key_code = cv2.waitKey(1) & 0xFF
                if key_code & 0xFF == 27:
                    break



cap.release()

# terminate all processes
for p in multiprocessing.active_children():
    p.terminate()
    p.join()
