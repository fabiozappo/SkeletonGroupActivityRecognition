import copy
import time
import argparse

import numpy as np

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

import dataset
from Configs import Config, features_clustering

from Models.GarSkeletonModel import GarSkeletonModel


def adjust_learning_rate(optimizer, epoch, lr, epochs_for_decay):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // epochs_for_decay))
    print('Changing learning rate in ', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return optimizer


def compute_confusion_matrix(test_loader, model, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for inputs, group_labels, distances, num_actors, persons_labels_batch in test_loader:
            # Move parameters to gpu, num_actors is just for iterating. Need to be on cpu.
            inputs, distances, group_labels, persons_labels_batch = inputs.to(Config.device), distances.to(
                Config.device), group_labels.to(Config.device), persons_labels_batch.to(Config.device)

            since = time.time()
            persons_output_list, group_output = model(inputs, distances, num_actors)
            _, preds = torch.max(group_output, 1)
            print('Predicting time for mini-batch: ', time.time() - since)
            for t, p in zip(group_labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print('Confusion Matrix:', confusion_matrix.long())
    print('Class accuracies:', confusion_matrix.diag() / confusion_matrix.sum(1))
    print('MCA:', (confusion_matrix.diag().sum() / confusion_matrix.sum()))
    print('MPCA:', torch.mean(confusion_matrix.diag() / confusion_matrix.sum(1)))

    return (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()


def train_model(model, dataloaders_dict, criterion_single, criterion_groups, optimizer, num_epochs, lr,
                epochs_for_decay, lr_decay, loss_factor):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_group_acc = 0.0
    indiv_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if lr_decay and (epoch % epochs_for_decay == 0):
            adjust_learning_rate(optimizer, epoch, lr, epochs_for_decay)

        # Each epoch has a training and validation phase
        for phase in ['trainval', 'test']:
            if phase == 'trainval':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_indiv_corrects = 0
            running_group_corrects = 0

            # Iterate over data.
            for inputs, group_labels, distances, num_actors, persons_labels_batch in dataloaders_dict[phase]:
                # Move parameters to gpu, num_actors is just for iterating. Need to be on cpu.
                inputs, distances, group_labels, persons_labels_batch = inputs.to(Config.device), distances.to(
                    Config.device), group_labels.to(Config.device), persons_labels_batch.to(Config.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'trainval'):
                    # Get model outputs and calculate group activity loss, criterion for crossentropy elements reduction over mini_batch is 'sum'
                    persons_output_list, group_output = model(inputs, distances, num_actors)
                    loss_group = criterion_groups(group_output, group_labels)

                    _, group_preds = torch.max(group_output, 1)

                    # Computing individual action loss, can't be vectorized due to variable number of actors
                    loss_persons = []
                    for persons_outputs, persons_labels in zip(persons_output_list, persons_labels_batch):
                        loss_persons.append(
                            criterion_single(persons_outputs, persons_labels[:persons_outputs.size()[0]]))
                        _, indiv_preds = torch.max(persons_outputs, 1)
                        running_indiv_corrects += torch.sum(indiv_preds == persons_labels[:persons_outputs.size()[
                            0]].data)  # updating individual action accuracy of the whole dataset

                    total_loss = loss_group + loss_factor * sum(loss_persons)

                    # backward + optimize only if in training phase
                    if phase == 'trainval':
                        total_loss.backward()
                        optimizer.step()

                running_loss += total_loss.item()  # updating loss of the whole dataset
                running_group_corrects += torch.sum(
                    group_preds == group_labels.data)  # updating group accuracy of the whole dataset

            # Epoch statistics to display
            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_individual_acc = running_indiv_corrects.double() / dataloaders_dict[phase].dataset.get_num_actors()
            epoch_group_acc = running_group_corrects.double() / len(dataloaders_dict[phase].dataset)

            # Display Epoch Results
            print('{} Loss  {:.4f}'.format(phase, epoch_loss))
            print('{} Accuracy: Individual {:.4f}, Groups {:.4f}'.format(phase, epoch_individual_acc, epoch_group_acc))

            # deep copy the model
            if phase == 'test' and epoch_group_acc > best_group_acc:
                best_group_acc = epoch_group_acc
                indiv_acc = epoch_individual_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_group_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_group_acc.item(), indiv_acc.item()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pseudo_labels', action='store_true', help='cluster action labels from visual features')
    parser.add_argument('--num_clusters', type=int, default=20, help='number of clusters')
    parser.add_argument('--augment', action='store_true', help='use horizontal flip augmentation')
    parser.add_argument('--epochs', type=int, default=80, help='number of training epochs')
    parser.add_argument('--batch_size', default=64, help='batch size dimension')
    parser.add_argument('--workers', type=int, default=8, help='num_workers')
    parser.add_argument('--loss_balancer', type=float, default=0.2, help='factor between individual and group loss')

    args = parser.parse_args()

    # Define double loss and balance factor
    loss_factor = 0.2
    criterion_single = nn.CrossEntropyLoss(reduction='sum')
    criterion_groups = nn.CrossEntropyLoss(reduction='sum')

    lr_activity = 0.001
    lr_decay = True
    epoch_decay = 30

    num_classes = args.num_clusters if args.pseudo_labels else Config.num_action_classes

    if args.pseudo_labels:
        visual_features = {phase: features_clustering.compute_visual_features(phase) for phase in ['trainval', 'test']}
        pca_model = features_clustering.fit_pca(256, visual_features)
        pca_features = {phase: features_clustering.compute_pca_features(visual_features[phase], pca_model) for phase in ['trainval', 'test']}
        kmeans_trained = features_clustering.fit_kmeans(args.num_clusters, pca_features)

        group_datasets = {phase: dataset.GroupFeatures(phase, kmeans_trained=kmeans_trained, pca_features=pca_features,
                                                       augment=args.augment, pseudo_labels=args.pseudo_labels) for phase in ['trainval', 'test']}
    else:
        group_datasets = {phase: dataset.GroupFeatures(phase, augment=args.augment, pseudo_labels=args.pseudo_labels)
                          for phase in ['trainval', 'test']}

    # Create training and test dataloaders
    dataloaders_dict = {
        'trainval': DataLoader(group_datasets['trainval'], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers),
        'test': DataLoader(group_datasets['test'], batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    }

    myNet = GarSkeletonModel(num_action_classes=num_classes)
    myNet.to(Config.device)
    print(myNet)

    # Loss and optimizer
    optimizer = torch.optim.Adam(myNet.parameters(), lr=lr_activity)

    # Train and evaluate, save best accuracy
    model_ft, acc_groups, acc_persons = train_model(myNet, dataloaders_dict, criterion_single, criterion_groups,
                                                    optimizer,
                                                    args.epochs, lr_activity, epoch_decay, lr_decay,
                                                    loss_factor=args.loss_balancer)

    # Confusion Matrix
    compute_confusion_matrix(dataloaders_dict['test'], model_ft, Config.num_group_activity_classes)

    np.set_printoptions(precision=4)
    print('Best Acc Persons: ', acc_persons)
    print('Best Acc Groups: ', acc_groups)
    print('\n')
