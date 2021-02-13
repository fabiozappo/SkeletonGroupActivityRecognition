import torch

# Common Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
has_to_apply_smoothing = True
has_to_erase_feet_and_head = True
batch_size = 128
remove_precision_feature = False  # todo: da implementare
normalize_feature = True # todo: da sistemare
mean_torso_len = 33.86
mean_midhip = [42.9451, 68.7730]

# Baselines
useEarlyFusion = False

# Action Classification Pose and Motion Parameters
useBalancedBatchesTrain = False
useBalancedBatchesTest = False
use_weighted_loss = False
num_epochs_action = 40 # todo
net_type = 'NTU'  # todo: serve dare una sistematina anche alla sbu (versione ridotta)
lr_action = 0.001
lr_decay_action = True
epochs_for_decay_action = 20  # todo: change in 10
num_action_classes = 9

# Group Activity Classification Parameters
num_group_activity_classes = 8
num_epochs_activity = 30 # todo
lr_decay_activity = True
epochs_for_decay_activity = 10
lr_activity = 0.001
max_pooling_group_features = True

# Work not in progress
has_to_compute_group_features_respect_to_pivot = True  # todo: andrebbe fatto di default
input_dim = 15 * 15 * 2
hidden_dim = 256  # todo: lstm verra' riusata un giorno?
layer_dim = 1  # todo: lstm verra' riusata un giorno?
seq_dim = 10  # todo: lstm verra' riusata un giorno?
use_dist_as_input_stream = True

# Unsupervised clustering
use_pseudo_labels = False # TODO: RICORDATI CHE KMEANS E PCA ORA SONO DETERMINISTICI
range_clusters = range(10, 91, 10) # todo 101
num_runs_kmeans = 1
test_visual_features = True

#End to End model
use_end_to_end_model = True
use_double_loss_model = True