import dlib
import os
from collections import deque
import numpy as np
import glob
import cv2
import json
from tqdm import tqdm
from shutil import copyfile
import glob

video_folder = '/work/sk-gar/volleyball_dataset/videos'
activity_list = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_winpoint', 'l-pass', 'l-spike', 'l_set']
action_list = ['waiting', 'setting', 'digging', 'falling', 'spiking', 'blocking', 'jumping', 'moving', 'standing']
max_num_players = 12
num_frames = 16
bbox_margin_gain = 0.02

# Create the correlation tracker - the object needs to be initialized before it can be used


def get_persons():
    for video_match in tqdm(os.listdir(video_folder)):
        annotation_file = os.path.join(video_folder, video_match, 'annotations.txt')
        if os.path.exists(annotation_file):
            with open(annotation_file) as f:
                lines = f.readlines()
            imgs = {}

            # each line contains the annotated frame of the 41 frames video sequence
            for line in lines:

                frame_id, rects = extract_info_from_annotation_line(line)
                # img_list = sorted(glob.glob(os.path.join(video_folder, video_match, frame_id, "*.jpg")))[16:26]
                img_list = glob.glob(os.path.join(video_folder, video_match, frame_id, "*.jpg"))
                img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))[20 - int((num_frames - 1) / 2):21 + int(np.ceil(float(num_frames - 1) / 2.0))]
                imgs['pre'] = img_list[:8][::-1]
                imgs['back'] = img_list[7:]

                # print('\n', img_list)

                assert len(
                    rects) <= max_num_players, 'TROVATO FRAME CON PIU DI 12 PERSONE'  # NB non tutti i frame annotati hanno 12 bb
                track(rects, imgs, frame_id)
        else:
            print('annotation_file not found:', annotation_file)  # TODO: copy from some source


def extract_info_from_annotation_line(line):
    keywords = deque(line.strip().split(' '))  # 'r-pass 318 690 102' ---> 'r-pass', '318', '690', '102'
    frame_id = keywords.popleft().split('.')[0]  # 31950.jpg l_winpoint 1370 650 59 98 ---> 31950
    activity = activity_list.index(keywords.popleft())

    rects = []
    while keywords:
        x = int(keywords.popleft())
        y = int(keywords.popleft())
        w = int(keywords.popleft())
        h = int(keywords.popleft())

        action = action_list.index(keywords.popleft())
        rects.append([x, y, w, h, action, activity])

    rects = np.asarray(rects)
    # sort Rects by the first col
    rects = rects[np.lexsort(rects[:, ::-1].T)]
    return frame_id, rects


def track(person_rects, imgs, frame_id):
    tracker = dlib.correlation_tracker()

    for i, person_rect in enumerate(person_rects):
        for j, phase in enumerate(['pre', 'back']):
            if j == 0:
                j = -1
            for k, f in enumerate(imgs[phase]):

                frame = cv2.imread(f)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x, y, w, h, action_label, activity_label = person_rect

                # If it's the first
                if k == 0:
                    # This object will start tracking the thing inside the bounding box in the given image.
                    tracker.start_track(frame_rgb, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))

                else:
                    # If you call update() with subsequent video frames then it will try to keep track of the position
                    # of the object inside bounding_box.
                    tracker.update(frame_rgb)

                pos = tracker.get_position()

                if not (k == 0 and j == -1):  # just write one time central frame
                    save_cropped_image(f, i, frame, pos, action_label, activity_label)


def save_cropped_image(frame_path, person_index, frame, pos, action_label, activity_label):

    h, w, _ = frame.shape
    top, bottom, left, right = pos.top(), pos.bottom(), pos.left(), pos.right()
    top, bottom = np.clip(int(top * (1-bbox_margin_gain)), 0, h), np.clip(int(bottom * (1+bbox_margin_gain)), 0, h)
    left, right = np.clip(int(left * (1-bbox_margin_gain)), 0, w), np.clip(int(right * (1+bbox_margin_gain)), 0, w)

    save_path = frame_path.replace('videos', 'tracked_persons').replace('.jpg', '/')

    if not (os.path.exists(save_path)):
        os.makedirs(save_path)

    save_img_path = f'{save_path + str(person_index)}_{top}_{left}_{bottom}_{right}_{action_label}_{activity_label}.jpg'

    cropped_image = frame[top:bottom, left:right]

    try:
        cv2.imwrite(save_img_path, cropped_image)
    except:
        print(save_img_path)  # '/work/sk-gar/volleyball_dataset/tracked_persons/11/32835/32843/10_380_1280_502_1280_8_3.jpg
        split_path = save_img_path.split('/')[:-2]  # /work/sk-gar/volleyball_dataset/tracked_persons/11/32835/
        current_folder = save_img_path.split('/')[-2]
        previous_folder = str(int(current_folder)-1)  # 32843 ----> 32842
        idx_actor = save_img_path.split('/')[-1].split('_')[0]
        previous_frame = glob.glob('/'.join(split_path + [previous_folder, idx_actor+'_*.jpg']))[0].split('/')[-1]  # 10_356_1235_517_1280_8_3.jpg
        src = '/'.join(split_path + [previous_folder, previous_frame])
        dst = '/'.join(split_path + [current_folder, previous_frame])
        copyfile(src, dst)
        print(f'Failed to write image, probably due to zero bb dimension: {save_img_path}, \nUsing previous frame in tracklet: {src}')


# Main
get_persons()
