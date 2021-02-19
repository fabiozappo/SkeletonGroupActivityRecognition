import torch

# Common Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
has_to_apply_smoothing = True
has_to_erase_feet_and_head = True
normalize_feature = True
mean_torso_len = 33.86 # todo
mean_midhip = [42.9451, 68.7730] # todo

# Action Classification Pose and Motion Parameters
num_action_classes = 9

# Group Activity Classification Parameters
num_group_activity_classes = 8

# Work not in progress
has_to_compute_group_features_respect_to_pivot = True
use_dist_as_input_stream = True
