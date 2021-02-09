import dlib
import os
from collections import deque
import numpy as np
import glob
import cv2
import json


video_folder = '/delorean/fzappardino/dataset/VD/videos'
save_folder_annotations = '/delorean/fzappardino/dataset/VD/augmented_annotations'
save_folder_images_with_bb = '/delorean/fzappardino/dataset/VD/augmented_images_bb'
activity_list = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_winpoint', 'l-pass', 'l-spike', 'l_set']
action_list = ['waiting', 'setting', 'digging', 'falling', 'spiking', 'blocking', 'jumping', 'moving', 'standing']
max_num_players = 12

# Create the correlation tracker - the object needs to be initialized before it can be used

def get_persons():

    for video_match in os.listdir(video_folder):

        annotation_file = os.path.join(video_folder, video_match, 'annotations.txt')
        if os.path.exists(annotation_file):
            f = open(annotation_file)
            lines = f.readlines()
            imgs = {}

            # each line contains the annotated frame of the 41 frames video sequence
            for line in lines:
                frame_id, rects = extract_info_from_annotation_line(line)
                # img_list = sorted(glob.glob(os.path.join(video_folder, video_match, frame_id, "*.jpg")))[16:26]  # TODO: in questo modo loro prendono i 4 prima e i 5 dopo! Ibrahim suggerisce il contratrio!
                img_list = sorted(glob.glob(os.path.join(video_folder, video_match, frame_id, "*.jpg")))[13:29]  # TODO: in questo modo loro prendono i 4 prima e i 5 dopo! Ibrahim suggerisce il contratrio!
                # imgs['pre'] = img_list[:5][::-1]
                imgs['pre'] = img_list[:8][::-1]
                # imgs['back'] = img_list[4:]
                imgs['back'] = img_list[7:]

                assert len(rects) <= max_num_players, 'TROVATO FRAME CON PIU DI 12 PERSONE' # NB non tutti i frame annotati hanno 12 bb
                track(rects, imgs, frame_id)  # TODO: comment to use opencv tracker


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
        rects.append([x, y, w, h, action, activity])  # TODO: serve nel nostro caso la label dell'activity?

    rects = np.asarray(rects)
    # sort Rects by the first col
    rects = rects[np.lexsort(rects[:, ::-1].T)]  # TODO: a che serve ordinarli per l'ascissa ???
    return frame_id, rects


def track(person_rects, imgs, frame_id):
    tracker = dlib.correlation_tracker()

    for i, person_rect in enumerate(person_rects):
        for j, phase in enumerate(['pre', 'back']):
            if j == 0:
                j = -1
            for k, f in enumerate(imgs[phase]):

                # frame_img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                frame_img = cv2.imread(f)
                x, y, w, h, label, group_label = person_rect

                # If it's the first
                if k == 0:
                    # This object will start tracking the thing inside the bounding box in the given image.
                    tracker.start_track(frame_img, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))

                else:
                    # If you call update() with subsequent video frames then it will try to keep track of the position of the object inside bounding_box.
                    tracker.update(frame_img)


                pos = tracker.get_position()
                top, bottom, left, right = max(int(pos.top()), 0), max(int(pos.bottom()), 0), max(int(pos.left()),0), max(int(pos.right()), 0)


                if not (k == 0 and j == -1): # serve a non scrivere due volte l'annotazione di riferimento
                    print f
                    save_cropped_image(f, int(frame_id)+(j*k), i, frame_img, top, left, bottom, right, label, group_label )


def draw_bb_from_openpose_joints():
    for subdir, dirs, files in os.walk('/delorean/fzappardino/openpose_results/VD/jsons/13'):

        for file in files:
            json_dir = os.path.join(subdir,file)
            img_read_dir = json_dir.replace('jsons','imgs').replace('keypoints.json','rendered.png')
            img_save_dir = img_read_dir.replace('imgs', 'openpose_rect_imgs')

            if not os.path.exists(subdir.replace('jsons','openpose_rect_imgs')):
                os.makedirs(subdir.replace('jsons','openpose_rect_imgs'))

            with open(json_dir) as json_file:
                data = json.load(json_file)
                img = cv2.imread(img_read_dir)
                for person in data['people']:
                    joints = np.array(person['pose_keypoints_2d']).reshape(25,3)
                    joints = joints[~np.all(joints == 0, axis=1)]

                    right, bottom = np.amax(joints, axis=0)[:2]
                    left, top = np.amin(joints, axis=0)[:2]

                    img = cv2.rectangle(img, (int(left*0.98), int(top*0.98)), (int(right*1.02), int(bottom*1.02)), (0, 0, 255), 2)

                    # print top, left, bottom, right

                cv2.imwrite(img_save_dir, img)


def save_cropped_image(frame_path, frame_number, person_index, img, top, left, bottom, right, action_label, activity_label, tracked = True ):

    top = int(top * 0.98)
    left = int(left * 0.98)
    bottom = int(bottom * 1.02)
    right = int(right * 1.02)


    save_path = frame_path.replace('videos','person_cropped_imgs_original_dimensions_DLIB').replace('.jpg','/')

    if not (os.path.exists(save_path)):
        os.makedirs(save_path)
        print 'Cartelle create: ', save_path

    # /delorean/fzappardino/dataset/VD/person_imgs_..../match/sequence
    save_img_path = save_path + str(person_index) + '_' + str(frame_number) + '_' + str(action_label) + '_' + str(activity_label) + '.jpg'

    if not tracked: # TODO: just for try
        save_img_path = save_img_path.replace('.jpg', '_FAIL.jpg')

    print (save_img_path)

    cropped_image = img[top:bottom, left:right]

    cv2.imwrite(save_img_path, cropped_image)

#Main
get_persons();
