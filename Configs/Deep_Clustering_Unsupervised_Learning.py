from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
import time
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler, normalize
from torch import nn
from torchvision import models, transforms
from PIL import Image
# import Config
from sklearn.decomposition import PCA


# features_path = '/delorean/fzappardino/toremovealexnetfeatures'
features_path = '/delorean/fzappardino/toremovevgg16features'
dim_features = 4096

def compute_labels_try_try(mode, kmeans, pca_features): # versione dove si introduce il feed randomico
    if mode not in ['trainval', 'test']:
        raise ValueError("Invalid mode type. Expected one trainval or test")
    unsupervised_labels = kmeans.predict(pca_features[mode])

    # print 'unsupervised_labels of {} has shape {}, minimum value of {} and {}'.format(mode, unsupervised_labels.shape, min(unsupervised_labels), max(unsupervised_labels))
    # return torch.from_numpy(unsupervised_labels).long() # commented to return a list for easier action padding
    # print list(unsupervised_labels)
    return list(unsupervised_labels.astype(long))

def compute_labels_try(mode, num_clusters, pca_features): # versione dove non si ricaricano le pca features ogni volta
    if mode not in ['trainval', 'test']:
        raise ValueError("Invalid mode type. Expected one trainval or test")
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, precompute_distances=True)
    kmeans = kmeans.fit(pca_features['trainval'])
    unsupervised_labels = kmeans.predict(pca_features[mode])

    # print 'unsupervised_labels of {} has shape {}, minimum value of {} and {}'.format(mode, unsupervised_labels.shape, min(unsupervised_labels), max(unsupervised_labels))
    return torch.from_numpy(unsupervised_labels).long()

def compute_labels(mode, num_clusters, images_paths):
    if mode not in ['trainval', 'test']:
        raise ValueError("Invalid mode type. Expected one trainval or test")
    visual_features = compute_visual_features(mode, images_paths)
    alexnet_pca_features = {phase: compute_pca_features(phase, visual_features) for phase in ['trainval', 'test']}
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, precompute_distances=True, random_state=0)
    kmeans = kmeans.fit(alexnet_pca_features['trainval'])
    unsupervised_labels = kmeans.predict(alexnet_pca_features[mode])

    # print 'unsupervised_labels of {} has shape {}, minimum value of {} and {}'.format(mode, unsupervised_labels.shape, min(unsupervised_labels), max(unsupervised_labels))
    return torch.from_numpy(unsupervised_labels).long()

def get_alexenet_model():
    model = models.alexnet(pretrained=True)
    # remove last fully-connected layer
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    model.classifier = new_classifier
    print (model)
    return model

def get_vgg_model():
    model = models.vgg16(pretrained=True)
    print (model)
    # remove last fully-connected layer
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    model.classifier = new_classifier
    print (model)
    return model

def fit_kmeans(num_clusters, pca_features):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, precompute_distances=True)
    return kmeans.fit(pca_features['trainval'])

def fit_pca(num_components, visual_features):
    pca = PCA(n_components=num_components, whiten=True, svd_solver='full')
    return pca.fit(visual_features['trainval'])


def compute_visual_features(mode, images_paths=None):
    # filename = features_path + mode + '_try.pt'  # todo: in real clustering remove try and delete files
    filename = features_path + mode + '.pt'
    if os.path.exists(filename):  # todo: change 'or' in 'and' to compute feature anyway (ATTENTO A NON CANCELLARE LE FEATURES COMPLETE!)
        print ('Loading vgg features of shape')
        return torch.load(filename).detach().numpy()  # serve il .float().to(conf.device) ????
    else:
        # Compute features with pretrained model
        num_features = len(images_paths)
        visual_features = torch.zeros((num_features, dim_features), dtype=torch.float)

        model = get_vgg_model()
        model.eval().to('cpu').float()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for f in range(num_features):
            central_img_path = images_paths[f][4]
            input_image = Image.open(central_img_path)
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            with torch.no_grad():
                visual_features[f, :] = model(input_batch.to('cpu'))
            print ('Computing feature {}/{} '.format(f, num_features))

        torch.save(visual_features, filename)
        visual_features = visual_features.numpy()

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
        print (filename, 'exists!')
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
    visual_features = {phase: compute_visual_features(phase) for phase in ['trainval', 'test']}
    pca_model = fit_pca(256, visual_features)
    pca_features = {phase: compute_pca_features(phase, pca_model) for phase in ['trainval', 'test']}

    # labels = {phase: compute_labels(phase, 50, None) for phase in ['trainval', 'test']}





