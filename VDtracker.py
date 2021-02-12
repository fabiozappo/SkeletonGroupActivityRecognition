import dlib
import os
from collections import deque
import numpy as np
import glob
import cv2
import json
from tqdm import tqdm

video_folder = '/work/data_and_extra/volleyball_dataset/videos'
activity_list = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_winpoint', 'l-pass', 'l-spike', 'l_set']
action_list = ['waiting', 'setting', 'digging', 'falling', 'spiking', 'blocking', 'jumping', 'moving', 'standing']
max_num_players = 12
num_frames = 10
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
                imgs['pre'] = img_list[:5][::-1]
                imgs['back'] = img_list[4:]

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

                frame_img = cv2.imread(f)
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                x, y, w, h, action_label, activity_label = person_rect

                # If it's the first
                if k == 0:
                    # This object will start tracking the thing inside the bounding box in the given image.
                    tracker.start_track(frame_img, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))

                else:
                    # If you call update() with subsequent video frames then it will try to keep track of the position
                    # of the object inside bounding_box.
                    tracker.update(frame_img)

                pos = tracker.get_position()
                top, bottom, left, right = max(int(pos.top()), 0), max(int(pos.bottom()), 0), \
                                           max(int(pos.left()), 0), max(int(pos.right()), 0)

                if not (k == 0 and j == -1):  # just write one time central frame
                    save_cropped_image(f, i, frame_img, top, left, bottom, right, action_label, activity_label)


def save_cropped_image(frame_path, person_index, img, top, left, bottom, right, action_label, activity_label):

    top = int(top * (1-bbox_margin_gain))
    left = int(left * (1-bbox_margin_gain))
    bottom = int(bottom * (1+bbox_margin_gain))
    right = int(right * (1+bbox_margin_gain))

    save_path = frame_path.replace('videos', 'tracked_persons').replace('.jpg', '/')

    if not (os.path.exists(save_path)):
        os.makedirs(save_path)

    # save_img_path = save_path + str(person_index) + '_' + str(frame_number) + '_' + str(action_label) + '_' + str(
    #     activity_label) + '.jpg'

    save_img_path = f'{save_path + str(person_index)}_{top}_{left}_{bottom}_{right}_{action_label}_{activity_label}.jpg'

    cropped_image = img[top:bottom, left:right]
    cv2.imwrite(save_img_path, cropped_image)


# Main
get_persons()
