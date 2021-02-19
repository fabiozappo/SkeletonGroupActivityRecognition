import json
import random
import os
import time
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from scipy import ndimage
from Configs import Config, features_clustering
from sklearn.metrics.cluster import normalized_mutual_info_score
import glob
from tqdm import tqdm


# Trainval and test list
# trainval_and_test_dicts = {'trainval': [0, 1, 2, 3, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 36, 38, 39, 40, 41, 42, 46, 48, 49, 50, 51, 52, 53, 54],
#                            'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]}
trainval_and_test_dicts = {'trainval': [13],
                           'test': [23]}


action_list = ['waiting', 'setting', 'digging', 'falling', 'spiking', 'blocking', 'jumping', 'moving', 'standing']
activity_list = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_winpoint', 'l-pass', 'l-spike', 'l_set']


persons_imgs_path = '/work/sk-gar/volleyball_dataset/tracked_persons/'
skeletons_path = '/work/sk-gar/volleyball_dataset/tracked_skeletons/'
weights_path = '/work/sk-gar/Weights/p3d_rgb_199.checkpoint.pth.tar'
features_path = '/work/sk-gar/volleyball_dataset/P3Dfeatures'


def center_skeleton_in_midhip_and_divide_by_torso(group_spatio_temporal_feature):
    for p in range(group_spatio_temporal_feature.shape[0]):
        for t in range(group_spatio_temporal_feature.shape[3]):
            # mid_hip
            mid_hip = group_spatio_temporal_feature[p, 8, :2, t].copy()
            mid_hip = mid_hip if mid_hip.any() else Config.mean_midhip
            # torso len
            has_torso = group_spatio_temporal_feature[p, 8, :2, t].any() and group_spatio_temporal_feature[p, 1, :2, t].any()
            torso_len = np.linalg.norm(group_spatio_temporal_feature[p, 8, :2, t] - group_spatio_temporal_feature[p, 1, :2, t]) \
                        if has_torso else Config.mean_torso_len

            for j in range(group_spatio_temporal_feature.shape[1]):
                joint = group_spatio_temporal_feature[p, j, :, t]
                if joint.any():
                    group_spatio_temporal_feature[p, j, :2, t] = joint[:2] - mid_hip
                    group_spatio_temporal_feature[p, j, :2, t] = group_spatio_temporal_feature[p, j, :2, t] / torso_len

    return group_spatio_temporal_feature


def imgpath_from_nppath(json_path):
    # print json_path
    match_folder, window_folder, frame_folder, json_name = json_path.split('/')[-4:]
    image_name = json_name.replace('.npy', '.jpg')
    person_frame_path = os.path.join(skeletons_path, match_folder, window_folder, frame_folder, image_name)
    return person_frame_path


def flip_group_horizontally(group_spatio_temporal_feature, group_temporal_bbox, match):
    frame_w = 1920 if match in [2, 37, 38, 39, 40, 41, 44, 45] else 1280

    for p in range(group_spatio_temporal_feature.shape[0]):
        for t in range(group_spatio_temporal_feature.shape[3]):

            person_skeleton = group_spatio_temporal_feature[p, :, :, t]
            person_bbox = group_temporal_bbox[p, :, t]

            person_bbox[[1, 3]] = frame_w - person_bbox[[3, 1]]

            w = person_bbox[3] - person_bbox[1]
            valid_joints = np.all(person_skeleton, axis=1)
            person_skeleton[valid_joints, 0] = w - person_skeleton[valid_joints, 0]

    return group_spatio_temporal_feature, group_temporal_bbox


def erase_feet_and_head(group_feature):
    return np.delete(group_feature, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24], axis=1)


def load_skeleton(j_p):

    person_index, top, left, bottom, right, action_label, activity_label = j_p.split('/')[-1].split('.')[0].split('_')
    person_skeleton = np.load(j_p)
    person_bbox = np.array([int(top), int(left), int(bottom), int(right)])

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


def compute_pivot_in_group(group_bboxes):
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
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def compute_pivot_distances(group_spatio_temporal_feature, group_bboxes, pivot_index, match):
    # tre casi: uno, due o nessun joint mancante ----> differenza delle box, differenza bb, e differenza normale. Il pivot viene contato come tutti zeri!
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

                # Videos with resolution of 1920x1080. All others are 1280x720.
                if match in [2, 37, 38, 39, 40, 41, 44, 45]:
                    frame_dim = [1920, 1080]
                else:
                    frame_dim = [1280, 720]
                movements[p, j1, :, t] = distance_vector / frame_dim

    return movements


def initialize_group_feature_and_label_list(mode, skeletons_path):
    # Lists to return
    person_images_path = []  # each element is a 10 len image-path representing a single tube for an actor
    group_features_list = []
    group_action_labels_list = []
    group_activity_labels_list = []
    group_boxes = []
    match_folders = []

    for match_folder in tqdm(trainval_and_test_dicts[mode]):  # tracked_skeletons/0
        match_path = os.path.join(skeletons_path, str(match_folder))

        for seq_folder in os.listdir(match_path):  # tracked_skeletons/0/3596
            seq_path = os.path.join(match_path, seq_folder)
            main_frame_path = os.path.join(seq_path, seq_folder)  # tracked_skeletons/0/3596/3596

            number_of_actors = len(os.listdir(main_frame_path))
            group_spatio_temporal_feature = np.zeros((number_of_actors, 25, 3, 10))
            group_bb = np.zeros((number_of_actors, 4, 10))
            group_paths = np.empty((number_of_actors, 10), dtype=object)

            # To display Info
            group_actions = []  # contiene le n azioni degli attori, rilevante solo per display

            # Sorting actors by x position is relevant for concatenate them, and no relevant for maxpooling
            for p, jf_main in enumerate(sorted(os.listdir(main_frame_path), key=lambda x: int(x.split('_')[0]))):
                main_frame_folder = main_frame_path.split('/')[-2]
                person_index, top, left, bottom, right, action_label, activity_label = jf_main.split('.')[0].split('_')

                for t in range(-4, 6):  # tracked_skeletons/0/3596/3592/0_412_184_571_251_8_2.npy
                    frame_folder = str(int(main_frame_folder) + t)
                    np_path = glob.glob(os.path.join(seq_path, frame_folder, person_index + '_*.npy'))[0]
                    group_spatio_temporal_feature[p, :, :, t + 4], group_bb[p, :, t + 4] = load_skeleton(np_path)
                    group_paths[p, t + 4] = imgpath_from_nppath(np_path)

                group_actions.append(int(action_label))

            # all skeletons in group activity are ready
            if Config.has_to_erase_feet_and_head:
                group_spatio_temporal_feature = erase_feet_and_head(group_spatio_temporal_feature)
            if Config.has_to_apply_smoothing:
                group_spatio_temporal_feature = compute_smoothed_skeletons(group_spatio_temporal_feature)

            # Create also the person dataset
            for p in range(number_of_actors):
                person_images_path.append(group_paths[p, :])

            group_features_list.append(group_spatio_temporal_feature)

            group_activity_labels_list.append(int(activity_label))
            group_action_labels_list.append(group_actions)
            group_boxes.append(group_bb)
            match_folders.append(match_folder)

    dict = {
        'group_features_list': group_features_list,
        'group_activity_labels_list': group_activity_labels_list,
        'group_action_labels_list': group_action_labels_list,
        'person_images_path': person_images_path,
        'group_boxes': group_boxes,
        'match_folders': match_folders
    }

    for k, item in dict.items():
        print(f'{k} has len: {len(item)}')

    return dict


def nan_or_inf(tensor):
    return torch.isnan(tensor).any() or tensor.eq(float('inf')).any() or tensor.eq(float('-inf')).any()


# Inizializzazione comune delle features
features = {phase: initialize_group_feature_and_label_list(phase, skeletons_path) for phase in ['trainval', 'test']}
for phase in ['trainval', 'test']:
    if not os.path.exists(features_path + phase + '.npy'):
        features_clustering.compute_visual_features(phase, weights_path=weights_path, images_paths=features[phase]['person_images_path'])


class GroupFeatures(Dataset):

    def __init__(self, mode, kmeans_trained=None, pca_features=None, augment=False, pseudo_labels=False):
        if mode not in ['trainval', 'test']:
            raise ValueError("Invalid mode type. Expected one trainval or test")

        self.augment = mode == 'trainval' and augment

        self.group_features_list = features[mode]['group_features_list']
        self.labels_list = features[mode]['group_activity_labels_list']
        self.group_boxes = features[mode]['group_boxes']
        self.match_folders = features[mode]['match_folders']

        if pseudo_labels:
            cluster_indices = kmeans_trained.predict(pca_features[mode])
            self.action_labels = []
            start_index = 0
            for actor_actions in features[mode]['group_action_labels_list']:
                num_actors = len(actor_actions)
                self.action_labels.append(cluster_indices[start_index:start_index + num_actors].tolist())
                start_index += num_actors
        else:
            self.action_labels = features[mode]['group_action_labels_list']

        self.num_actors = sum([len(action_labels) for action_labels in self.action_labels])

    def __getitem__(self, index):

        group_skeleton = self.group_features_list[index]
        activity_label = self.labels_list[index]
        group_bb = self.group_boxes[index]
        match_folder = self.match_folders[index]
        action_labels = self.action_labels[index]

        if self.augment:
            if random.random() < 0.5:
                group_skeleton, group_bb = flip_group_horizontally(group_skeleton, group_bb, match_folder)
                activity_label = 7 - int(activity_label)

        pivot_index = compute_pivot_in_group(group_bb)
        pivot_distances = compute_pivot_distances(group_skeleton, group_bb, pivot_index, match_folder)

        if Config.normalize_feature:
            group_skeleton = center_skeleton_in_midhip_and_divide_by_torso(group_skeleton)

        group_skeleton = torch.from_numpy(group_skeleton.transpose(0, 2, 1, 3)).float()
        pivot_distances = torch.from_numpy(pivot_distances.transpose(0, 2, 1, 3)).float()

        max_actors = 12
        num_actors = group_skeleton.size()[0]

        padded_group_skeleton = torch.zeros([max_actors, 3, 15, 10], dtype=torch.float)
        padded_group_skeleton[:num_actors, :, :, :] = group_skeleton
        padded_group_distances = torch.zeros([max_actors, 2, 15, 10], dtype=torch.float)
        padded_group_distances[:num_actors, :, :, :] = pivot_distances
        padded_group_action_labels = np.array(action_labels[:num_actors] + [-1] * (max_actors - num_actors), dtype=int)

        assert not nan_or_inf(padded_group_skeleton), 'nan or inf value in padded_group_skeleton'
        assert not nan_or_inf(padded_group_distances), 'nan or inf value in padded_group_distances'

        return padded_group_skeleton, activity_label, padded_group_distances, num_actors, padded_group_action_labels

    def __len__(self):
        return len(self.labels_list)

    def get_num_actors(self):
        return self.num_actors


if __name__ == "__main__":
    # Create training and validation datasets
    group_datasets = {phase: GroupFeatures(phase) for phase in ['trainval', 'test']}
