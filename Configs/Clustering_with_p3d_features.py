# import cv2
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
import time
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler, normalize
# from torch import nn
from torchvision import models, transforms
from PIL import Image
from Configs import Config
from sklearn.decomposition import PCA
from Models.P3D import P3D199
from sklearn.manifold import TSNE

# features_path = '/delorean/fzappardino/toremovealexnetfeatures'

# features_path = '/delorean/fzappardino/P3Dfeatures'
features_path = '/work/data_and_extra/P3Dfeatures'
dim_features = 2048

def compute_labels_try_try(mode, kmeans, pca_features): # versione dove si introduce il feed randomico
    if mode not in ['trainval', 'test']:
        raise ValueError("Invalid mode type. Expected one trainval or test")
    unsupervised_labels = kmeans.predict(pca_features[mode])

    # print 'unsupervised_labels of {} has shape {}, minimum value of {} and {}'.format(mode, unsupervised_labels.shape, min(unsupervised_labels), max(unsupervised_labels))
    # return torch.from_numpy(unsupervised_labels).long() # commented to return a list for easier action padding
    # print list(unsupervised_labels)
    return list(unsupervised_labels.astype(long))

def get_P3D_model():
    print ('Loading P3D model...')
    model = P3D199(pretrained=True, num_classes=400)
    # alredy removed last fully-connected layer to do feature extraction
    model = model.to(Config.device)
    return model


def fit_kmeans(num_clusters, features):
    # kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, precompute_distances=True, random_state=0)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, precompute_distances=True, random_state=None)
    return kmeans.fit(features['trainval'])

def fit_pca(num_components, visual_features):
    pca = PCA(n_components=num_components, whiten=True)
    # pca = PCA(n_components=num_components, whiten=True, svd_solver='full')
    return pca.fit(visual_features['trainval'])


def compute_visual_features(mode, images_paths=None):
    filename = features_path + mode + '.npy'  # todo: in real clustering remove try and delete files
    # filename = features_path + mode + '.pt'
    if os.path.exists(filename):  # todo: change 'or' in 'and' to compute feature anyway (ATTENTO A NON CANCELLARE LE FEATURES COMPLETE!)
        print ('Loading ', filename, ' features!')
        return np.load(filename)
    else:
        # Compute features with pretrained model
        num_features = len(images_paths)
        visual_features = torch.zeros((num_features, dim_features), dtype=torch.float)

        model = get_P3D_model()
        model.eval().to(Config.device).float()

        preprocess = transforms.Compose([
            # transforms.Resize(224),
            transforms.Resize(160),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for f in range(num_features):
            clip = torch.zeros((1, 3, 16, 160, 160), dtype=torch.float).to(Config.device)

            original_image_path = images_paths[f][0]  # take random element
            path_list = original_image_path.split('/')
            seq_folder, frame_folder, img_name = path_list[-3:]
            path_list[5] = 'person_cropped_imgs_original_dimensions_DLIB_C3D'
            # print '\n'
            for t in range(-7, 9):
                frame_folder_number = str(int(seq_folder) + t)
                path_list[-1] = img_name.replace(frame_folder, frame_folder_number)
                path_list[-2] = frame_folder_number

                image_path = '/'.join(path_list)

                # print '{} frame of clip. loading {}'.format(t, image_path)
                input_image = Image.open(image_path)
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0)
                clip[:, :, t + 7, :, :] = input_batch

            clip = clip.contiguous()

            with torch.no_grad():
                visual_features[f, :] = model(clip)

            print ('Computing feature {}/{}, it will be saved at {} '.format(f, num_features, filename))

        visual_features = visual_features.numpy()
        np.save(filename, visual_features)


    return visual_features

def train_kmeans(X):
    wcss = []
    since = time.time()

    my_range = range(5, 100, 5)
    for i in my_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, precompute_distances=True)
        kmeans.fit(X)
        wcc = kmeans.inertia_
        wcss.append(wcc)
        print ('Centroid # {}, error of {}'.format(i, wcc))

    print ('time elapsed in learning clusters: ', time.time() - since)
    plt.plot(my_range, wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    # plt.savefig('/delorean/fzappardino/elbow_method_kmeans')


    val, idx = min((val, idx) for (idx, val) in enumerate(wcss))
    print ('minimum mean_squared_error = {} for k* = {} '.format(val, idx+my_range[0]))

    return idx+my_range[0]


def compute_pca_features(phase, pca_model=None):
    # filename = features_path + phase + '_pca_try.npy'
    # filename = features_path + phase + '_pca_whitened.npy'
    filename = features_path + phase + '_pca.npy'

    if os.path.exists(filename) and False:
        print( filename, 'exists!')
        principal_components = np.load(filename)
        print ('Loadad pca features of shape', principal_components.shape)
    else:
        visual_features = compute_visual_features(phase)
        print ('Computing PCA!')
        # Standardizing the features
        # x = StandardScaler().fit_transform(features)

        principal_components = pca_model.transform(visual_features)
        principal_components = normalize(principal_components)

        print ('input shape: {} -----> {}'.format(visual_features.shape, principal_components.shape))

        # np.save(filename, principal_components)

    return principal_components


if __name__ == "__main__":
    # '/delorean/fzappardino/dataset/VD/person_cropped_imgs_original_dimensions_DLIB/4/48980/48976/0_48976_8_2.jpg
    visual_features = {phase: compute_visual_features(phase) for phase in ['trainval', 'test']}
    pca_model = fit_pca(256, visual_features)
    pca_features = {phase: compute_pca_features(phase, pca_model) for phase in ['trainval', 'test']}

    # labels = {phase: compute_labels(phase, 50, None) for phase in ['trainval', 'test']}

    X = np.concatenate((pca_features['trainval'], pca_features['test']), axis=0)
    print ('X has shape: ', X.shape)
    tsne_results = TSNE(n_components=2, verbose=1).fit_transform(X)
    print ('tsne_results:', tsne_results.shape)

    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    # plt.show()

    df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])
    df_tsne['label'] = y[rows[:n_select]]
    sns.lmplot(x='comp1', y='comp2', data=df_tsne, hue='label', fit_reg=False)



