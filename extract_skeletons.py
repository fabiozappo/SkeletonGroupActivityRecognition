import logging
import argparse
import time
import cv2
import numpy as np
import pyopenpose as op
import os
import pickle
from pathlib import Path

from tqdm import trange

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="/work/sk-gar/volleyball_dataset/",
                    help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--save_skeletons", action='store_true', help="Enable to save skeletons.")
parser.add_argument("--no_display", action='store_true', help="Enable to disable the visual display.")
args = parser.parse_args()


def frame_is_in_clip(frame_path):
    central_frame = int(frame_path.split('/')[-3])
    frame = int(frame_path.split('/')[-2])
    return central_frame-4 <= frame <= central_frame+5


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict(model_folder="/openpose/models/", number_people_max=1)
# params["net_resolution"] = '368x368' # uncomment this line if you have low gpu memory

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


data_path = Path(args.image_dir)
annot = data_path / 'tracks.pkl'
# read BB annotation file
with open(annot, "rb") as file:
    annot_data = pickle.load(file)

# Read frames on directory
image_list = list(data_path.glob("**/*.jpg"))
image_iter = trange(len(image_list))

# Process and display images
for im in image_iter:
    image_path = image_list[im]
    print(image_path)
    video_id = int(image_path.parent.parent.stem)
    frame_id = int(image_path.parent.stem)
    image_id = int(image_path.stem)
    if image_id not in annot_data[(video_id, frame_id)].keys():
        continue

    # Read frame just one time
    imageToProcess = cv2.imread(str(image_path))

    for i, bbox in enumerate(annot_data[(video_id, frame_id)][image_id]):
        out_file = image_path.parent / f'{image_id}_{i}.npy'
        if not out_file.exists():
            datum = op.Datum()

            # Crop BB from image
            crop = imageToProcess[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()

            datum.cvInputData = crop
            opWrapper.emplaceAndPop([datum])

            joints = datum.poseKeypoints
            np_joints = np.array(joints).squeeze(axis=0)  # assert single person detection

            if not args.no_display:
                plot_image = datum.cvOutputData

                if np_joints.shape == (25, 3):
                    for x, y, p in np_joints[:15]:
                        plot_image = cv2.circle(plot_image, (x, y), radius=2, color=(0, 0, 0), thickness=-1)
                cv2.imshow("OpenPose Results", plot_image)
                cv2.waitKey(0)

            if args.save_skeletons:
                out_file = str(out_file)
                out_folder = '/'.join(out_file.split('/')[:-1])
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)

                # if joint extraction fails save np.zeros(25x3) instead of np.empty()
                np_joints = np_joints if np_joints.shape == (25, 3) else np.zeros((25, 3))
                np.save(out_file, np_joints)
