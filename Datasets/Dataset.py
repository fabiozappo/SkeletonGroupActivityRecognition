import json
import os
import time
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from scipy import ndimage
from Configs import Config, Deep_Clustering_Unsupervised_Learning, Clustering_with_p3d_features
from sklearn.metrics.cluster import normalized_mutual_info_score


hasToShowImages = False
hasToUseAbsoluteCoordinates = False  # todo: forse sarebbe piu opportuno chiamarlo hasToShowAbsoluteSkeletons?
hasToShowDistances, person_to_show_dist = False, 0

# Trainval and test list
# trainval_and_test_dicts = {'trainval': [0, 1, 2, 3, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 36, 38, 39, 40, 41, 42, 46, 48, 49, 50, 51, 52, 53, 54],
#                            'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]}
trainval_and_test_dicts = {'trainval': [0],
                           'test': [2]}

action_list = ['waiting', 'setting', 'digging', 'falling', 'spiking', 'blocking', 'jumping', 'moving', 'standing']
activity_list = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_winpoint', 'l-pass', 'l-spike', 'l_set']

POSE_PAIRS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
              (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23),
              (11, 24)]

REDUCED_POSE_PAIRS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12),
                      (12, 13), (13, 14), (1, 0)]

original_local_imgs_path = '/home/fabio/Scrivania/dati/person_cropped_imgs_original_dimensions_DLIB/'
original_remote_imgs_path = '/delorean/fzappardino/dataset/VD/person_cropped_imgs_original_dimensions_DLIB/'


# Group dataset: 3490 gruppi di training
# Group dataset: 1337 gruppi di test


def center_skeleton_in_midhip_and_divide_by_torso(group_spatio_temporal_feature):
    for p in range(group_spatio_temporal_feature.shape[0]):
        for t in range(group_spatio_temporal_feature.shape[3]):
            mid_hip = group_spatio_temporal_feature[p, 8, :2, t] if group_spatio_temporal_feature[p, 8, :2,
                                                                    t].any() else Config.mean_midhip
            torso = group_spatio_temporal_feature[p, 8, :2, t] - group_spatio_temporal_feature[p, 1, :2,
                                                                 t] if group_spatio_temporal_feature[p, 8, :2,
                                                                       t].any() and group_spatio_temporal_feature[p, 1,
                                                                                    :2,
                                                                                    t].any() else Config.mean_torso_len
            torso_len = np.linalg.norm(torso)
            # print '\nmid_hip and torso', mid_hip, torso_len
            for j in range(group_spatio_temporal_feature.shape[1]):
                joint = group_spatio_temporal_feature[p, j, :, t]
                # print 'original joint :', joint
                if joint.any():
                    group_spatio_temporal_feature[p, j, :2, t] = joint[:2] - mid_hip
                    # print 'shifted joint :', spatio_temporal_feature[j, :, t]
                    group_spatio_temporal_feature[p, j, :2, t] = group_spatio_temporal_feature[p, j, :2, t] / torso_len
                    # print 'shifted and normalized joint :', spatio_temporal_feature[j, :, t]
    return group_spatio_temporal_feature


def recover_imgpath_from_jsonpath(json_path):
    # print json_path
    match_folder, window_folder, frame_folder, json_name = json_path.split('/')[-4:]
    image_name = json_name.replace('_keypoints.json', '.jpg')
    person_frame_path = os.path.join(original_remote_imgs_path, match_folder, window_folder, frame_folder, image_name)
    # print person_frame_path, os.path.exists(person_frame_path)
    return person_frame_path


def recover_img_from_jsonpath(json_path):
    # print json_path
    match_folder, window_folder, frame_folder, json_name = json_path.split('/')[-4:]
    image_name = json_name.replace('_keypoints.json', '.jpg')
    person_frame_path = os.path.join(original_local_imgs_path, match_folder, window_folder, frame_folder, image_name)
    # print person_frame_path
    return cv2.imread(person_frame_path)


def recover_global_img_from_jsonpath(jf_path, flipped):
    # print 'reading json:', jf_path
    global_frame_path = '/'.join(jf_path.replace('fixed_jsons', 'Volleyball_Frames').split('/')[:-1]) + '.jpg'
    print('global frame path:', global_frame_path)
    originalImage = cv2.imread(global_frame_path)
    return cv2.flip(originalImage, 1) if flipped else originalImage


def fill_frames_list(global_frames, group_bboxes, person_frames, jf_path, p, t, flipped=False):
    if hasToUseAbsoluteCoordinates:
        if p == 0:
            print('reading image from json: ', jf_path)
            global_frames.append(recover_global_img_from_jsonpath(jf_path, flipped))
    else:
        person_frames.append(recover_img_from_jsonpath(jf_path))


def show_person_on_global_frame(group_spatio_temporal_feature, global_frames, activity_label, group_actions,
                                group_bboxes, pivot_index, movements_respect_to_pivot):
    central_bbox = get_central_bbox(group_bboxes)

    for t in range(group_spatio_temporal_feature.shape[3]):
        global_frame = global_frames[t]
        cv2.rectangle(global_frame, (int(central_bbox[0]), int(central_bbox[1])),
                      (int(central_bbox[2]), int(central_bbox[3])), (255, 255, 255), 1,
                      lineType=cv2.LINE_AA)

        for p in range(group_spatio_temporal_feature.shape[0]):
            skeleton = group_spatio_temporal_feature[p, :, :, t]
            action_label = group_actions[p]
            points = []
            bbox = group_bboxes[p, :, t]

            cv2.putText(global_frame, action_list[int(action_label)], (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0),
                        1, lineType=cv2.LINE_AA)
            color = (0, 255, 255) if p == pivot_index else (0, 0, 0)
            cv2.rectangle(global_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1,
                          lineType=cv2.LINE_AA)

            for x, y in zip(skeleton[:, 0], skeleton[:, 1]):
                if x != 0 and y != 0:
                    x += bbox[0]
                    y += bbox[1]
                    points.append((int(x), int(y)))
                    cv2.circle(global_frame, (int(x), int(y)), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                else:
                    points.append((None, None))
            # Draw Skeleton
            PAIRS = REDUCED_POSE_PAIRS if Config.has_to_erase_feet_and_head else POSE_PAIRS
            for partA, partB in PAIRS:
                if all(points[partA]) and all(points[partB]):
                    cv2.line(global_frame, points[partA], points[partB], (0, 255, 0), 2)

            if hasToShowDistances and person_to_show_dist == p:
                # todo: il joint di un attore e' la coda della freccia, il joint piu' la distanza e' la coda (expected pivot joint)
                pivot_bbox = group_bboxes[pivot_index, :2, t]
                abs_pivot = group_spatio_temporal_feature[pivot_index, :, :2, t] + pivot_bbox
                abs_skeleton = skeleton[:, :2] + bbox[:2]

                for i, (sj, pj) in enumerate(zip(abs_skeleton[:, :2], abs_pivot[:, :2])):
                    distance_from_pj = movements_respect_to_pivot[p, i * 2, t], movements_respect_to_pivot[
                        p, i * 2 + 1, t]
                    expected_pj = sj + distance_from_pj

                    # print '{} skeleton joint + distance_vector expected = pivot_joint, {} + {} = {} ?= {}'.format(i, sj, distance_from_pj, expected_pj, pj)
                    if (sj - bbox[:2]).any() and (pj - pivot_bbox[:2]).any():
                        cv2.arrowedLine(global_frame, (int(sj[0]), int(sj[1])),
                                        (int(expected_pj[0]), int(expected_pj[1])), (0, 0, 0), 1)  # expected
                    else:
                        cv2.arrowedLine(global_frame, (int(bbox[0]), int(bbox[1])),
                                        (int(pivot_bbox[0]), int(pivot_bbox[1])), (0, 0, 0), 1)  # expected

        winname = 'global-frame, activity:{}'.format(activity_list[int(activity_label)])
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, 40, 30)
        cv2.imshow(winname, global_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_skeletons_on_images(group_descriptor, group_frames, activity_label, group_actions):
    for p in range(len(group_frames)):
        person_descriptor = group_descriptor[p, :, :, :]
        person_frames = group_frames[p]
        action_label = group_actions[p]
        shape = person_frames[0].shape
        for i in range(person_descriptor.shape[2]):
            skeleton = person_descriptor[:, :, i]
            person_image = person_frames[i]
            points = []
            cv2.putText(person_image, action_list[int(action_label)], (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0),
                        1, lineType=cv2.LINE_AA)
            for x, y in zip(skeleton[:, 0], skeleton[:, 1]):
                if x != 0 and y != 0:
                    points.append((int(x), int(y)))
                    cv2.circle(person_image, (int(x), int(y)), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                else:
                    points.append((None, None))
            # # Draw Skeleton
            PAIRS = REDUCED_POSE_PAIRS if Config.has_to_erase_feet_and_head else POSE_PAIRS
            for partA, partB in PAIRS:
                if all(points[partA]) and all(points[partB]):
                    cv2.line(person_image, points[partA], points[partB], (0, 255, 0), 2)
            if i == 0:
                concatenated_images = person_image
            else:
                resized = cv2.resize(person_image, (shape[1], shape[
                    0]))  # todo: anche i joints andrebbero poi spostati, ma sull'immagine non si nota la differenza di pochi px!
                concatenated_images = np.concatenate((concatenated_images, resized), axis=1)

        cv2.imshow(
            'Tube of person {}/{} in activity {}'.format(p, len(group_frames), activity_list[int(activity_label)]),
            concatenated_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def flip_horizontally(person_skeleton, person_bbox, match):
    flipped_skeleton = np.copy(person_skeleton)
    flipped_bbox = np.copy(person_bbox)

    if match in [2, 37, 38, 39, 40, 41, 44, 45]:
        flipped_bbox[[2, 0]] = [1920, 1920] - person_bbox[[0, 2]]
    else:
        flipped_bbox[[2, 0]] = [1280, 1280] - person_bbox[[0, 2]]

    for i in range(person_skeleton.shape[0]):
        if person_skeleton[i, :].any():
            flipped_skeleton[i, 0] = (person_skeleton[i, 0] - (person_bbox[2] - person_bbox[0]) / 2.0) * (-1.0) + (
                    (person_bbox[2] - person_bbox[
                        0]) / 2.0)  # todo: provare semplicemente: person_skeleton[i, 0] = (person_bbox[2] - person_bbox[0]) - person_skeleton[i, 0]

    return flipped_skeleton, flipped_bbox


def erase_feet_and_head(group_feature):
    return np.delete(group_feature, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24], axis=1)


def compute_descriptor_from_json_keypoints(j_p):  # todo: dovrebbe restituire numpy_array e bb
    with open(j_p) as json_file:
        loaded_json = json.load(json_file)
        print(j_p)
        people = loaded_json['people']
        person_bbox = loaded_json['bbox']
        person_bbox = np.asarray(person_bbox)

        if len(people) == 1:
            person_skeleton = np.array(people[0]['pose_keypoints_2d']).reshape(25, 3)
        else:
            person_skeleton = np.zeros((25, 3))

        return person_skeleton, person_bbox


def compute_smoothed_skeletons(skeletons_group):
    for p in range(skeletons_group.shape[0]):
        for i in range(skeletons_group.shape[2]):
            for j in range(skeletons_group.shape[1]):
                joint_series = skeletons_group[p, j, i, :]
                if joint_series.any():
                    # Compute the mean value of the time-series of the joint, missing values are not considered
                    mean = np.mean(joint_series[joint_series != 0])
                    original_and_mean_joints = np.where(joint_series == 0, mean, joint_series)
                    # Use average filter to smooth missing joints
                    smoothed_joints = ndimage.correlate1d(original_and_mean_joints,
                                                          weights=[1. / 14, 2. / 14, 4. / 14, 0, 4. / 14, 2. / 14,
                                                                   1. / 14], mode='nearest')
                    skeletons_group[p, j, i, :] = np.where(original_and_mean_joints == mean, smoothed_joints,
                                                           original_and_mean_joints)
    return skeletons_group


def compute_pivot_in_group(group_spatio_temporal_feature, group_bboxes):
    central_bbox = get_central_bbox(group_bboxes)
    best_iou = 0.0
    best_index = 5
    for p in range(group_bboxes.shape[0]):
        actor_bbox_in_central_frame = group_bboxes[p, :, 4]
        iou = bb_intersection_over_union(actor_bbox_in_central_frame, central_bbox)
        if iou > best_iou:
            best_index = p
            best_iou = iou
    return best_index


def get_central_bbox(group_bboxes):
    return np.mean(group_bboxes, axis=0)[:, 4]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compute_movements_respect_to_pivot_joint_joint_forNtuInput(group_spatio_temporal_feature, group_bboxes, pivot_index,
                                                               match):  # todo: prova per return di un vettore 15x10
    # tre casi: uno, due o nessun joint mancante ----> differenza delle box, differenza bb,  e differenza normale. Il pivot viene contato come tutti zeri!
    shape = group_spatio_temporal_feature.shape
    movements = np.zeros((shape[0], 15, 2, shape[3]))  # group movements represented as array of size (n, 225, 10)
    pivot = group_spatio_temporal_feature[pivot_index, :, :, :]

    for p in range(group_spatio_temporal_feature.shape[0]):  # loop over persons in group
        for t in range(group_spatio_temporal_feature.shape[3]):  # loop over time
            person_bbox = group_bboxes[p, :, t]
            pivot_bbox = group_bboxes[pivot_index, :, t]

            distance_between_bboxes = pivot_bbox[:2] - person_bbox[:2]

            for j1 in range(group_spatio_temporal_feature.shape[1]):  # loop over joint
                target_joint = group_spatio_temporal_feature[p, j1, :2, t] + person_bbox[:2]  # target_joint.shape = [2]
                pivot_joint = pivot[j1, :2, t] + pivot_bbox[:2]

                if pivot_joint.any() and target_joint.any():
                    distance_vector = (pivot_joint - target_joint).flatten()
                else:
                    distance_vector = distance_between_bboxes.flatten()

                if match in [2, 37, 38, 39, 40, 41, 44,
                             45]:  # Videos with resolution of 1920x1080 are: 2 37 38 39 40 41 44 45 (8 in total). All others are 1280x720.
                    frame_dim = [1920, 1080]
                else:
                    frame_dim = [1280, 720]
                movements[p, j1, :, t] = distance_vector / frame_dim

    return movements.transpose(0, 2, 1, 3)  # NTU net requires joint coordinates in channel dim


def initialize_group_feature_and_label_list(mode, skeletons_path):
    # Lists to return
    person_features_list = []
    person_labels_list = []
    person_group_dynamic_features = []  # lista di differenze di scheletri 15 x 2 x 10
    person_images_path = []  # each element is a 10 len image-path representing a single tube for an actor

    flipped_person_features_list = []
    flipped_person_group_dynamic_features = []
    flipped_person_labels_list = []
    flipped_group_features_list = []
    flipped_group_group_dynamic_features = []
    flipped_group_activity_labels_list = []
    flipped_group_action_labels_list = []

    group_features_list = []
    group_action_labels_list = []
    group_activity_labels_list = []
    group_group_dynamic_features = []  # lista di differenze di scheletri n x 15 x 2 x 10
    group_images_path = []  # contiene n, 10 stringhe image-path, un blocco = una sequenza, potrebbe servire????

    for match_folder in trainval_and_test_dicts[mode]:  # jsons/0
        match_path = os.path.join(skeletons_path, str(match_folder))
        print('loading jsons of match:', match_path)

        for seq_folder in os.listdir(match_path):  # jsons/0/3596
            seq_path = os.path.join(match_path, seq_folder)
            main_frame_path = os.path.join(seq_path, seq_folder)  # jsons/0/3596/3596

            number_of_actors = len(os.listdir(main_frame_path))
            # Group level info
            group_spatio_temporal_feature = np.zeros((number_of_actors, 25, 3, 10))
            flipped_group_spatio_temporal_feature = np.zeros((number_of_actors, 25, 3, 10))
            group_bboxes = np.zeros((number_of_actors, 4, 10))
            flipped_group_bboxes = np.zeros((number_of_actors, 4, 10))
            group_paths = np.empty((number_of_actors, 10), dtype=object)

            # To display Info
            group_frames = []  # contiene i tubi degli attori (lista di liste di frames)
            global_frames = []  # contiene i dieci frame della sequenza di match
            flipped_global_frames = []  # contiene i dieci frame della sequenza di match
            group_actions = []  # contiene le n azioni degli attori, rilevante solo per display

            # Sorting actors by x position is relevant for concatenate them, and no relevant for maxpooling
            for p, jf_main in enumerate(sorted(os.listdir(main_frame_path), key=lambda x: int(
                    x.split('_')[0]))):  # jsons/0/3596/3596/0_3596_8_2_keypoints.json
                person_index, main_frame_folder, action_label, activity_label = jf_main.split('/')[-1].split('_')[:-1]
                person_frames = []  # tubo dell'attore p

                for t in range(-4, 6):  # jsons/0/3596/3592/0_3596_8_2_keypoints.json
                    frame_folder = str(int(main_frame_folder) + t)
                    jf = '_'.join([person_index, frame_folder, action_label, activity_label, 'keypoints.json'])
                    jf_path = os.path.join(seq_path, frame_folder, jf)
                    group_spatio_temporal_feature[p, :, :, t + 4], group_bboxes[p, :,
                                                                   t + 4] = compute_descriptor_from_json_keypoints(
                        jf_path)
                    flipped_group_spatio_temporal_feature[p, :, :, t + 4], flipped_group_bboxes[p, :,
                                                                           t + 4] = flip_horizontally(
                        group_spatio_temporal_feature[p, :, :, t + 4], group_bboxes[p, :, t + 4], match_folder)
                    group_paths[p, t + 4] = recover_imgpath_from_jsonpath(jf_path)

                    if hasToShowImages:
                        fill_frames_list(global_frames, group_bboxes, person_frames, jf_path, p, t)
                        fill_frames_list(flipped_global_frames, group_bboxes, person_frames, jf_path, p, t,
                                         flipped=True)

                group_actions.append(int(action_label))
                if not hasToUseAbsoluteCoordinates:
                    group_frames.append(person_frames)

                flipped_activity_label = 7 - int(activity_label)
            # tutti gli scheletri del gruppo sono pronti
            # if not group_spatio_temporal_feature.any():
            #     print 'found an empty group on seq', seq_path

            if Config.has_to_erase_feet_and_head:
                group_spatio_temporal_feature = erase_feet_and_head(group_spatio_temporal_feature)
                flipped_group_spatio_temporal_feature = erase_feet_and_head(flipped_group_spatio_temporal_feature)
            if Config.has_to_apply_smoothing:
                group_spatio_temporal_feature = compute_smoothed_skeletons(group_spatio_temporal_feature)
                flipped_group_spatio_temporal_feature = compute_smoothed_skeletons(
                    flipped_group_spatio_temporal_feature)

            if Config.has_to_compute_group_features_respect_to_pivot:  # quasi sempre a True altrimenti si rompe
                pivot_index = compute_pivot_in_group(group_spatio_temporal_feature, group_bboxes)
                flipped_pivot_index = compute_pivot_in_group(flipped_group_spatio_temporal_feature,
                                                             flipped_group_bboxes)
                movements_respect_to_pivot = compute_movements_respect_to_pivot_joint_joint_forNtuInput(
                    group_spatio_temporal_feature, group_bboxes, pivot_index, match_folder)
                flipped_movements_respect_to_pivot = compute_movements_respect_to_pivot_joint_joint_forNtuInput(
                    flipped_group_spatio_temporal_feature, flipped_group_bboxes, flipped_pivot_index, match_folder)

            if hasToShowImages:  # todo: ha senso che siano mutuamente esclusivi?
                if hasToUseAbsoluteCoordinates:
                    show_person_on_global_frame(group_spatio_temporal_feature, global_frames, activity_label,
                                                group_actions, group_bboxes, pivot_index, movements_respect_to_pivot)
                    show_person_on_global_frame(flipped_group_spatio_temporal_feature, flipped_global_frames,
                                                flipped_activity_label, group_actions, flipped_group_bboxes,
                                                pivot_index, flipped_movements_respect_to_pivot)
                else:
                    show_skeletons_on_images(group_spatio_temporal_feature, group_frames, activity_label, group_actions)

            if Config.normalize_feature:
                group_spatio_temporal_feature = center_skeleton_in_midhip_and_divide_by_torso(
                    group_spatio_temporal_feature)
                flipped_group_spatio_temporal_feature = center_skeleton_in_midhip_and_divide_by_torso(
                    flipped_group_spatio_temporal_feature)

            # Create also the person dataset
            for p in range(number_of_actors):
                person_spatio_temporal_feature = group_spatio_temporal_feature[p, :, :, :]
                flipped_person_spatio_temporal_feature = flipped_group_spatio_temporal_feature[p, :, :, :]
                person_spatio_temporal_feature = torch.from_numpy(
                    person_spatio_temporal_feature.transpose(1, 0, 2)).float()
                flipped_person_spatio_temporal_feature = torch.from_numpy(
                    flipped_person_spatio_temporal_feature.transpose(1, 0, 2)).float()
                person_features_list.append(person_spatio_temporal_feature)
                flipped_person_features_list.append(flipped_person_spatio_temporal_feature)
                person_group_dynamic_features.append(torch.from_numpy(movements_respect_to_pivot[p, :, :]).float())
                flipped_person_group_dynamic_features.append(
                    torch.from_numpy(flipped_movements_respect_to_pivot[p, :, :]).float())
                person_images_path.append(group_paths[p, :])
            person_labels_list = person_labels_list + group_actions
            flipped_person_labels_list = flipped_person_labels_list + group_actions

            if Config.useEarlyFusion:  # feature of different size requires little more code
                group_spatio_temporal_feature = group_spatio_temporal_feature[:, :, :, 4]
                early_fusion_feature = np.zeros((12, 15 if Config.has_to_erase_feet_and_head else 25, 3))
                early_fusion_feature[:number_of_actors, :, :] = group_spatio_temporal_feature
                early_fusion_feature = early_fusion_feature.transpose(1, 2, 0)
                group_features_list.append(torch.from_numpy(early_fusion_feature).float())
            else:
                group_features_list.append(
                    torch.from_numpy(group_spatio_temporal_feature.transpose(0, 2, 1, 3)).float())
                flipped_group_features_list.append(
                    torch.from_numpy(flipped_group_spatio_temporal_feature.transpose(0, 2, 1, 3)).float())
                group_group_dynamic_features.append(torch.from_numpy(movements_respect_to_pivot).float())
                flipped_group_group_dynamic_features.append(
                    torch.from_numpy(flipped_movements_respect_to_pivot).float())
                group_images_path.append(group_paths)

            group_activity_labels_list.append(int(activity_label))
            flipped_group_activity_labels_list.append(int(flipped_activity_label))
            group_action_labels_list.append(group_actions)
            flipped_group_action_labels_list.append(group_actions)

    dict = {
        'group_features_list': group_features_list,
        'group_activity_labels_list': group_activity_labels_list,
        'group_group_dynamic_features': group_group_dynamic_features,
        'group_images_path': group_images_path,
        'group_action_labels_list': group_action_labels_list,
        'person_features_list': person_features_list,
        'person_group_dynamic_features': person_group_dynamic_features,
        'person_labels_list': person_labels_list,
        'person_images_path': person_images_path,

        'flipped_group_features_list': flipped_group_features_list,
        'flipped_group_group_dynamic_features': flipped_group_group_dynamic_features,
        'flipped_group_activity_labels_list': flipped_group_activity_labels_list,
        'flipped_group_action_labels_list': flipped_group_action_labels_list,
        'flipped_person_features_list': flipped_person_features_list,
        'flipped_person_group_dynamic_features': flipped_person_group_dynamic_features,
        'flipped_person_labels_list': flipped_person_labels_list,

    }

    print('person_features_list has len: {}, flipped has len: {}'.format(len(person_features_list),
                                                                         len(flipped_person_features_list)))
    print('person_group_dynamic_features has len: {}, flipped has len: {}'.format(len(person_group_dynamic_features),
                                                                                  len(
                                                                                      flipped_person_group_dynamic_features)))
    print('person_labels_list has len: {}, flipped has len: {}'.format(len(person_labels_list),
                                                                       len(flipped_person_labels_list)))
    print('group_features_list has len: {}, flipped has len: {}'.format(len(group_features_list),
                                                                        len(flipped_group_features_list)))
    print('group_group_dynamic_features has len: {}, flipped has len: {}'.format(len(group_group_dynamic_features), len(
        flipped_group_group_dynamic_features)))
    print('group_action_labels_list has len: {}, flipped has len: {}'.format(len(group_action_labels_list),
                                                                             len(flipped_group_action_labels_list)))
    print('group_activity_labels_list has len: {}, flipped has len: {}'.format(len(group_activity_labels_list),
                                                                               len(flipped_group_activity_labels_list)))

    return dict


# Inizializzazione comune delle features
since = time.time()
# skeletons_path = '/delorean/fzappardino/openpose_results/VD/person_cropped_original_dimensions_DLIB/fixed_jsons' if not hasToShowImages else '/home/fabio/Scrivania/dati/fixed_jsons'
skeletons_path = '/work/data_and_extra/fixed_jsons' if not hasToShowImages else '/home/fabio/Scrivania/dati/fixed_jsons'
features = {phase: initialize_group_feature_and_label_list(phase, skeletons_path) for phase in ['trainval', 'test']}

print('time elapsed in creating groups features dataset:', time.time() - since)


class PersonFeatures(Dataset):
    def __init__(self, mode, skeletons_path=None, num_clusters=None, nmi_values=None, i=None, j=None, pca_features=None,
                 kmeans_trained=None):
        if mode not in ['trainval', 'test']:
            raise ValueError("Invalid mode type. Expected one trainval or test")
        if mode == 'trainval':
            self.features_list = features[mode]['person_features_list'] + features[mode]['flipped_person_features_list']
            self.distance_features = features[mode]['person_group_dynamic_features'] + features[mode][
                'flipped_person_group_dynamic_features']
            self.labels_list = features[mode]['person_labels_list'] + features[mode]['flipped_person_labels_list']
        else:
            self.features_list = features[mode]['person_features_list']
            self.distance_features = features[mode]['person_group_dynamic_features']
            self.labels_list = features[mode]['person_labels_list']

        images_paths = features[mode]['person_images_path']
        if Config.test_visual_features:
            Clustering_with_p3d_features.compute_visual_features(mode,
                                                                 images_paths)  # todo COMMENTARE, SERVE SOLO PER CREARE LE FEATURES!
            self.visual_features = np.load(
                '/delorean/fzappardino/P3Dfeatures16' + mode + '_pca.npy')  # P3Dfeatures, toremovevgg16featurestest_pca
            # self.visual_features = self.visual_features[: (1783 if mode == 'trainval' else 1275), :] # todo: COMMENTARE se non e' un test_try

        if Config.use_pseudo_labels:  # todo: mettere condizione?
            self.unsupervised_labels = Deep_Clustering_Unsupervised_Learning.compute_labels_try_try(mode,
                                                                                                    kmeans_trained,
                                                                                                    pca_features)
            # self.unsupervised_labels = self.unsupervised_labels[: (1783 if mode == 'trainval' else 1275)] # todo: COMMENTARE se non e' un test_try

            np.set_printoptions(precision=4)
            print('Mode:', mode)
            print('Working with num clusters:', num_clusters, 'at run # ', i)
            # print 'Labels list:', self.labels_list
            # print 'Pseudo Labels list:', self.unsupervised_labels
            nmi_values[i, j] = normalized_mutual_info_score(self.labels_list, self.unsupervised_labels,
                                                            average_method='geometric')
            print('nmi value: ', nmi_values[i, j])

            nmi_values[i, j] = normalized_mutual_info_score(self.labels_list, self.fake_labels)

    def __getitem__(self, index):
        label = self.labels_list[index] if not Config.use_pseudo_labels else self.unsupervised_labels[index]
        # label = self.fake_labels[index]
        if Config.test_visual_features:
            return self.visual_features[index], label, self.distance_features[index]
        else:
            return self.features_list[index], label, self.distance_features[index]

    def __len__(self):
        return len(self.labels_list)


class GroupFeatures(Dataset):

    def __init__(self, mode, skeletons_path=None, pose_and_motion_model=None, distances_model=None, kmeans_trained=None,
                 pca_features=None):
        if mode not in ['trainval', 'test']:
            raise ValueError("Invalid mode type. Expected one trainval or test")
        self.model = pose_and_motion_model  # per adesso si lascia su cpu
        if mode == 'trainval':
            self.group_features_list = features[mode]['group_features_list'] + features[mode][
                'flipped_group_features_list']
            self.labels_list = features[mode]['group_activity_labels_list'] + features[mode][
                'flipped_group_activity_labels_list']
            self.distance_features = features[mode]['group_group_dynamic_features'] + features[mode][
                'flipped_group_group_dynamic_features']
            self.action_labels_list = features[mode]['group_action_labels_list'] + features[mode][
                'flipped_group_action_labels_list']
        else:
            self.group_features_list = features[mode]['group_features_list']
            self.labels_list = features[mode]['group_activity_labels_list']
            self.distance_features = features[mode]['group_group_dynamic_features']
            self.action_labels_list = features[mode]['group_action_labels_list']

        self.num_actors = sum([len(action_labels) for action_labels in self.action_labels_list])
        self.model_internal_feature_list = []
        # self.action_labels_list = [torch.randint(low=0, high=9, size=(12,)) for group_features in self.group_features_list]  # todo: remove random initialization

        if Config.use_pseudo_labels and Config.use_double_loss_model:
            unsupervised_labels = Clustering_with_p3d_features.compute_labels_try_try(mode, kmeans_trained,
                                                                                      pca_features)  # Clusterizza le features, che siano Vgg16 o P3D clustering
            # unsupervised_labels = unsupervised_labels[: (1783 if mode == 'trainval' else 1275)] # todo: remove after testing
            if mode == 'trainval':
                unsupervised_labels = unsupervised_labels * 2  # todo: just for data augmentation!

            # print 'phase: {}, nmi value: {}'.format(mode, normalized_mutual_info_score(features[mode]['person_labels_list'], unsupervised_labels, average_method='geometric'))
            self.pseudo_action_labels_list = []
            start_index = 0
            for actor_actions in self.action_labels_list:
                num_actors = len(actor_actions)
                self.pseudo_action_labels_list.append(unsupervised_labels[start_index:start_index + num_actors])
                # print 'pseudo_action_labels_list has len {} and last item has {} labels'.format(len(self.pseudo_action_labels_list), len(self.pseudo_action_labels_list[-1]))
                start_index += num_actors

        # print '{} group_features_list: {}'.format(mode, len(self.group_features_list))
        # print '{} labels_list: {}'.format(mode, len(self.labels_list))
        # print '{} distance_features: {}'.format(mode, len(self.distance_features))
        # print '{} num_actors: {}'.format(mode, self.num_actors)

    def __getitem__(self, index):
        if not Config.use_end_to_end_model:
            return self.model_internal_feature_list[index] if self.model else self.group_features_list[index], \
                   self.labels_list[index]
        else:  # use end to end model, just groups, padded teams
            max_num_actors_in_scene = 12
            num_actors = self.group_features_list[index].size()[0]
            padded_group_skeleton = torch.zeros([max_num_actors_in_scene, 3, 15, 10], dtype=torch.float)
            padded_group_skeleton[:num_actors, :, :, :] = self.group_features_list[index]
            padded_group_distances = torch.zeros([max_num_actors_in_scene, 2, 15, 10], dtype=torch.float)
            padded_group_distances[:num_actors, :, :, :] = self.distance_features[index]
            padded_group_action_labels = np.array(self.action_labels_list[index][:num_actors] + [-1] * (
                        max_num_actors_in_scene - len(self.action_labels_list[index])), dtype=int)
            if Config.use_pseudo_labels:
                padded_pseudo_group_action_labels = np.array(
                    self.pseudo_action_labels_list[index][:num_actors] + [-1] * (
                                max_num_actors_in_scene - len(self.pseudo_action_labels_list[index])), dtype=int)
                # print 'returning padded_pseudo_group_action_labels: ', padded_pseudo_group_action_labels
            if Config.use_double_loss_model:
                assert not (torch.isnan(padded_group_skeleton).any() or padded_group_skeleton.eq(
                    float('inf')).any() or padded_group_skeleton.eq(
                    float('-inf')).any()), 'nan or inf value in padded_group_skeleton'
                assert not (torch.isnan(padded_group_distances).any() or padded_group_distances.eq(
                    float('inf')).any() or padded_group_distances.eq(
                    float('-inf')).any()), 'nan or inf value in padded_group_distances'
                return padded_group_skeleton, self.labels_list[
                    index], padded_group_distances, num_actors, padded_group_action_labels if not Config.use_pseudo_labels else padded_pseudo_group_action_labels
            else:
                return padded_group_skeleton, self.labels_list[index], padded_group_distances

    def __len__(self):
        return len(self.labels_list)

    def get_num_actors(self):
        return self.num_actors

    # def get_num_teams(self):
    #     return len(self.action_labels_list)


if __name__ == "__main__":
    # Create training and validation datasets
    person_datasets = {phase: PersonFeatures(phase) for phase in ['trainval', 'test']}
    # group_datasets = {phase: GroupFeatures(phase) for phase in ['trainval', 'test']}
