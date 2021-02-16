import json
import os
import time
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from scipy import ndimage
from Configs import Config, Clustering_with_p3d_features
from sklearn.metrics.cluster import normalized_mutual_info_score
import glob
from tqdm import tqdm


# Trainval and test list
trainval_and_test_dicts = {'trainval': [0, 1, 2, 3, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 36, 38, 39, 40, 41, 42, 46, 48, 49, 50, 51, 52, 53, 54],
                           'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]}
# trainval_and_test_dicts = {'trainval': [13],
#                            'test': [23]}

action_list = ['waiting', 'setting', 'digging', 'falling', 'spiking', 'blocking', 'jumping', 'moving', 'standing']
activity_list = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_winpoint', 'l-pass', 'l-spike', 'l_set']


persons_imgs_path = '/work/data_and_extra/volleyball_dataset/tracked_persons/'
skeletons_path = '/work/data_and_extra/volleyball_dataset/tracked_skeletons/'
weights_path = '/work/code/Weights/p3d_rgb_199.checkpoint.pth.tar'
features_path = '/work/data_and_extra/volleyball_dataset/P3Dfeatures'


# Group dataset: 3490 gruppi di training
# Group dataset: 1337 gruppi di test


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
    image_name = json_name.replace('.npy', '.jpg')
    person_frame_path = os.path.join(skeletons_path, match_folder, window_folder, frame_folder, image_name)
    # print person_frame_path, os.path.exists(person_frame_path)
    return person_frame_path


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


def compute_movements_respect_to_pivot_joint_joint_forNtuInput(group_spatio_temporal_feature, group_bboxes, pivot_index, match):  # todo: prova per return di un vettore 15x10
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

                # Videos with resolution of 1920x1080. All others are 1280x720.
                if match in [2, 37, 38, 39, 40, 41, 44, 45]:
                    frame_dim = [1920, 1080]
                else:
                    frame_dim = [1280, 720]
                movements[p, j1, :, t] = distance_vector / frame_dim

    return movements.transpose(0, 2, 1, 3)  # NTU net requires joint coordinates in channel dim


def initialize_group_feature_and_label_list(mode, skeletons_path):
    # Lists to return
    person_images_path = []  # each element is a 10 len image-path representing a single tube for an actor

    flipped_group_features_list = []
    flipped_group_group_dynamic_features = []
    flipped_group_activity_labels_list = []
    flipped_group_action_labels_list = []

    group_features_list = []
    group_action_labels_list = []
    group_activity_labels_list = []
    group_group_dynamic_features = []  # lista di differenze di scheletri n x 15 x 2 x 10

    for match_folder in tqdm(trainval_and_test_dicts[mode]):  # jsons/0
        match_path = os.path.join(skeletons_path, str(match_folder))
        # print('loading jsons of match:', match_path)

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
            group_actions = []  # contiene le n azioni degli attori, rilevante solo per display

            # Sorting actors by x position is relevant for concatenate them, and no relevant for maxpooling
            for p, jf_main in enumerate(sorted(os.listdir(main_frame_path), key=lambda x: int(x.split('_')[0]))):  # jsons/0/3596/3596/0_3596_8_2_keypoints.json
                main_frame_folder = main_frame_path.split('/')[-2]
                person_index, top, left, bottom, right, action_label, activity_label = jf_main.split('.')[0].split('_')

                for t in range(-4, 6):  # jsons/0/3596/3592/0_3596_8_2_keypoints.json
                    frame_folder = str(int(main_frame_folder) + t)
                    jf = person_index + '_*.npy'
                    jf_path = glob.glob(os.path.join(seq_path, frame_folder, jf))[0]
                    group_spatio_temporal_feature[p, :, :, t + 4], group_bboxes[p, :, t + 4] = compute_descriptor_from_json_keypoints(jf_path)
                    flipped_group_spatio_temporal_feature[p, :, :, t + 4], flipped_group_bboxes[p, :, t + 4] = flip_horizontally(group_spatio_temporal_feature[p, :, :, t + 4], group_bboxes[p, :, t + 4], match_folder)
                    group_paths[p, t + 4] = recover_imgpath_from_jsonpath(jf_path)

                group_actions.append(int(action_label))

                flipped_activity_label = 7 - int(activity_label)
            # tutti gli scheletri del gruppo sono pronti
            if not group_spatio_temporal_feature.any():
                print ('found an empty group on seq', seq_path)

            if Config.has_to_erase_feet_and_head:  # todo: move in get item
                group_spatio_temporal_feature = erase_feet_and_head(group_spatio_temporal_feature)
                flipped_group_spatio_temporal_feature = erase_feet_and_head(flipped_group_spatio_temporal_feature)
            if Config.has_to_apply_smoothing:  # todo: move in get item
                group_spatio_temporal_feature = compute_smoothed_skeletons(group_spatio_temporal_feature)
                flipped_group_spatio_temporal_feature = compute_smoothed_skeletons(
                    flipped_group_spatio_temporal_feature)

            if Config.has_to_compute_group_features_respect_to_pivot:  # todo: move in get item
                pivot_index = compute_pivot_in_group(group_bboxes)
                flipped_pivot_index = compute_pivot_in_group(flipped_group_bboxes)
                movements_respect_to_pivot = compute_movements_respect_to_pivot_joint_joint_forNtuInput(
                    group_spatio_temporal_feature, group_bboxes, pivot_index, match_folder)
                flipped_movements_respect_to_pivot = compute_movements_respect_to_pivot_joint_joint_forNtuInput(
                    flipped_group_spatio_temporal_feature, flipped_group_bboxes, flipped_pivot_index, match_folder)

            if Config.normalize_feature:   # todo: move in get item
                group_spatio_temporal_feature = center_skeleton_in_midhip_and_divide_by_torso(group_spatio_temporal_feature)
                flipped_group_spatio_temporal_feature = center_skeleton_in_midhip_and_divide_by_torso(flipped_group_spatio_temporal_feature)

            # Create also the person dataset
            for p in range(number_of_actors):
                person_images_path.append(group_paths[p, :])

            group_features_list.append(torch.from_numpy(group_spatio_temporal_feature.transpose(0, 2, 1, 3)).float())
            flipped_group_features_list.append(torch.from_numpy(flipped_group_spatio_temporal_feature.transpose(0, 2, 1, 3)).float())
            group_group_dynamic_features.append(torch.from_numpy(movements_respect_to_pivot).float())
            flipped_group_group_dynamic_features.append(torch.from_numpy(flipped_movements_respect_to_pivot).float())

            group_activity_labels_list.append(int(activity_label))
            flipped_group_activity_labels_list.append(int(flipped_activity_label))
            group_action_labels_list.append(group_actions)
            flipped_group_action_labels_list.append(group_actions)

    dict = {
        'group_features_list': group_features_list,
        'group_activity_labels_list': group_activity_labels_list,
        'group_group_dynamic_features': group_group_dynamic_features,
        'group_action_labels_list': group_action_labels_list,
        'person_images_path': person_images_path,

        'flipped_group_features_list': flipped_group_features_list,  # todo: move in get item
        'flipped_group_group_dynamic_features': flipped_group_group_dynamic_features,  # todo: move in get item
        'flipped_group_activity_labels_list': flipped_group_activity_labels_list,  # todo: move in get item
        'flipped_group_action_labels_list': flipped_group_action_labels_list,

    }

    print('person_clip has len: {}'.format(len(person_images_path)))

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
features = {phase: initialize_group_feature_and_label_list(phase, skeletons_path) for phase in ['trainval', 'test']}
visual_features = {phase: Clustering_with_p3d_features.compute_visual_features(phase, weights_path=weights_path,
                   images_paths=features[phase]['person_images_path']) for phase in ['trainval', 'test']}

print('time elapsed in creating groups features dataset:', time.time() - since)



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
        self.person_clips = features[mode]['person_images_path']
        # self.action_labels_list = [torch.randint(low=0, high=9, size=(12,)) for group_features in self.group_features_list]  # todo: remove random initialization

        if Config.use_pseudo_labels and Config.use_double_loss_model:
            unsupervised_labels = Clustering_with_p3d_features.compute_labels_try_try(mode, kmeans_trained,
                                                                                      pca_features)  # Clusterizza le features, che siano Vgg16 o P3D clustering
            if mode == 'trainval':
                unsupervised_labels = unsupervised_labels * 2  # todo: just for data augmentation!

            self.pseudo_action_labels_list = []
            start_index = 0
            for actor_actions in self.action_labels_list:
                num_actors = len(actor_actions)
                self.pseudo_action_labels_list.append(unsupervised_labels[start_index:start_index + num_actors])
                # print 'pseudo_action_labels_list has len {} and last item has {} labels'.format(len(self.pseudo_action_labels_list), len(self.pseudo_action_labels_list[-1]))
                start_index += num_actors

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


if __name__ == "__main__":
    # Create training and validation datasets
    group_datasets = {phase: GroupFeatures(phase) for phase in ['trainval', 'test']}
