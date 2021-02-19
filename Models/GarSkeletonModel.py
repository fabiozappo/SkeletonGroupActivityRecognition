import torch.nn as nn
import torch

from Configs import Config
import torch.nn.functional as F


class GarSkeletonModel(nn.Module):
    def __init__(self, num_action_classes=Config.num_action_classes):
        super(GarSkeletonModel, self).__init__()
        self.num_joints = 15 if Config.has_to_erase_feet_and_head else 25
        self.device = Config.device
        self.concatenated_dim = 96 if Config.use_dist_as_input_stream else 64
        self.num_action_classes = num_action_classes  # varibile per pseudolabels.
        self.num_group_classes = Config.num_group_activity_classes
        self.input_size = 1024

        # position
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv3 = nn.Conv2d(in_channels=self.num_joints, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                   nn.MaxPool2d(2)
                                   )
        # motion
        self.conv1m = nn.Conv2d(3, 64, kernel_size=1, padding=0, stride=1)
        self.conv2m = nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv3m = nn.Conv2d(in_channels=self.num_joints, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4m = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                    nn.MaxPool2d(2)
                                    )
        # pivot distance
        self.conv1d = nn.Conv2d(2, 64, kernel_size=1, padding=0, stride=1)
        self.conv2d = nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv3d = nn.Conv2d(in_channels=self.num_joints, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4d = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                    nn.MaxPool2d(2)
                                    )
        # concatenate motion & position & pivot-differences
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.concatenated_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(  # feature extracted layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        # flatten part for predicting actor action
        self.fc7 = nn.Sequential(
            nn.Linear(1024, 256),  # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.fc8 = nn.Linear(256, self.num_action_classes)

        # flatten part for predicting collective activity
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, self.num_group_classes)

    def forward(self, X, D, N):

        # Tensor of dim: batch_size, 1024
        group_features = torch.zeros([X.size()[0], 1024], dtype=torch.float).to(Config.device)
        persons_output_list = []

        for b in range(X.size()[0]):
            num_actors = N[b]
            x = X[b, :num_actors, :, :, :].view(-1, 3, self.num_joints, 10)
            d = D[b, :num_actors, :, :, :].view(-1, 2, self.num_joints, 10)

            # position
            out = self.conv1(x)
            out = self.conv2(out)
            out = out.permute(0, 2, 1, 3).contiguous()
            out = self.conv3(out)
            out_p = self.conv4(out)

            # motion computed with vectorization and interpolation
            motion = x[:, :, :, 1::] - x[:, :, :, 0:-1]
            motion = F.interpolate(motion, size=(self.num_joints, 10), mode='bilinear', align_corners=False)
            out = self.conv1m(motion)
            out = self.conv2m(out)
            out = out.permute(0, 2, 1, 3).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)

            # pivot difference
            if Config.use_dist_as_input_stream:
                out = self.conv1d(d)
                out = self.conv2d(out)
                out = out.permute(0, 2, 1, 3).contiguous()
                out = self.conv3d(out)
                out_d = self.conv4d(out)

            # concatenate motion & position
            out = torch.cat((out_p, out_m, out_d), dim=1) if Config.use_dist_as_input_stream else torch.cat(
                (out_p, out_m), dim=1)
            out = self.conv5(out)
            out = self.conv6(out)

            # print 'out.size: ', out.size()

            # aggregating features at group level
            person_feas = out.view(out.size(0), -1)
            group_features[b, :] = torch.max(person_feas, 0)[0]

            # flatten part for classifying persons
            out = self.fc7(person_feas)
            out_person = self.fc8(out)  # size n, 9
            assert not ((out_person != out_person).any())  # find out nan in tensor
            persons_output_list.append(out_person)

        # Group Level Classifier Layer
        out = self.fc1(group_features)
        group_output = self.fc2(out)
        assert not ((group_output != group_output).any())  # find out nan in tensor

        return persons_output_list, group_output
