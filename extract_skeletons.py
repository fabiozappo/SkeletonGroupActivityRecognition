import argparse
import time
import cv2
import numpy as np
import pyopenpose as op
import glob
from tqdm import tqdm
import os

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="/work/sk-gar/volleyball_dataset/tracked_persons/",
                    help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--save_skeletons", action='store_true', help="Enable to save skeletons.")
parser.add_argument("--no_display", action='store_true', help="Enable to disable the visual display.")
args = parser.parse_args()


def frame_is_in_clip(frame_path):
    central_frame = int(frame_path.split('/')[-3])
    frame = int(frame_path.split('/')[-2])
    return central_frame-4 <= frame <= central_frame+5


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/openpose/models/"
params["number_people_max"] = 1
params["net_resolution"] = '368x368'

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Read frames on directory
imagePaths = glob.glob(args.image_dir + '/**/**/**/*.jpg')
imagePaths = [f for f in imagePaths if frame_is_in_clip(f)]

# Process and display images
for imagePath in tqdm(imagePaths):

    out_file = imagePath.replace('tracked_persons', 'tracked_skeletons').replace('.jpg', '.npy')
    if not os.path.exists(out_file):
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)

        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        joints = datum.poseKeypoints
        np_joints = np.array(joints).squeeze(axis=0)  # assert single person detection

        if not args.no_display:
            plot_image = datum.cvOutputData
            for x, y, p in np_joints[:15]:
                plot_image = cv2.circle(plot_image, (x, y), radius=2, color=(0, 0, 0), thickness=-1)
            cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", plot_image)
            cv2.waitKey(0)

        if args.save_skeletons:
            out_folder = '/'.join(out_file.split('/')[:-1])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            # if joint extraction fails save np.zeros(25x3) instead of np.empty()
            np_joints = np_joints if np_joints.shape == (25, 3) else np.zeros((25, 3))
            np.save(out_file, np_joints)
