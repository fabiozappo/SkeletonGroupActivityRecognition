from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import time
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler, normalize
from torchvision import models, transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from Models.P3D import P3D199
from Configs import Config
import glob

features_path = '/work/sk-gar/volleyball_dataset/P3Dfeatures'
dim_features = 2048


def get_P3D_model(weights_path):
    print('Loading P3D model...')
    model = P3D199(weights_path, pretrained=True, num_classes=400)
    # alredy removed last fully-connected layer to do feature extraction
    model = model.to(Config.device)
    return model


def fit_kmeans(num_clusters, features):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=None)
    return kmeans.fit(features['trainval'])


def fit_pca(num_components, visual_features):
    pca = PCA(n_components=num_components, whiten=True, svd_solver='full')
    return pca.fit(visual_features['trainval'])


def compute_visual_features(mode, weights_path=None, images_paths=None):
    filename = features_path + mode + '.npy'
    if os.path.exists(filename):
        print('Loading ', filename, ' features!')
        return np.load(filename)
    else:
        print(f'Extracting {mode} visual features...')

        # Compute features with pretrained model
        num_features = len(images_paths)
        visual_features = torch.zeros((num_features, dim_features), dtype=torch.float)

        model = get_P3D_model(weights_path)
        model.eval().to(Config.device).float()

        preprocess = transforms.Compose([
            # transforms.Resize(224),
            transforms.Resize(160),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for f in tqdm(range(num_features)):
            clip = torch.zeros((1, 3, 16, 160, 160), dtype=torch.float).to(Config.device)

            original_image_path = images_paths[f][0]  # take random element
            path_list = original_image_path.split('/')
            seq_folder, frame_folder, img_name = path_list[-3:]
            path_list[4] = 'tracked_persons'

            actor_id = img_name.split('_')[0]
            # print '\n'
            for t in range(-7, 9):
                frame_folder_number = str(int(seq_folder) + t)
                path_list[-1] = img_name.replace(frame_folder, frame_folder_number)
                path_list[-2] = frame_folder_number

                root = '/'.join(path_list[:-1])

                image_path = glob.glob(root + f'/{actor_id}_*.jpg')[0]

                input_image = Image.open(image_path)
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0)
                clip[:, :, t + 7, :, :] = input_batch

            clip = clip.contiguous()

            with torch.no_grad():
                visual_features[f, :] = model(clip)

        visual_features = visual_features.numpy()
        np.save(filename, visual_features)

    return visual_features


def compute_pca_features(visual_features, pca_model):
    print('Computing PCA!')

    # Standardizing the features
    # x = StandardScaler().fit_transform(visual_features)

    principal_components = pca_model.transform(visual_features)
    principal_components = normalize(principal_components)

    print('input shape: {} -----> {}'.format(visual_features.shape, principal_components.shape))

    return principal_components
