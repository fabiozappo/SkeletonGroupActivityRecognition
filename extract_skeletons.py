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
parser.add_argument("--image_dir", default="/work/code/Datasets/tracked_persons/",
                    help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--save_skeletons", action='store_true', help="Enable to save skeletons.")
parser.add_argument("--no_display", action='store_true', help="Enable to disable the visual display.")
args = parser.parse_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/openpose/models/"
params["number_people_max"] = 1
params["net_resolution"] = '368x368' # todo: non mi garba


# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Read frames on directory
imagePaths = glob.glob(args.image_dir + '/**/**/**/*.jpg')

# Process and display images
for imagePath in tqdm(imagePaths):

    out_file = imagePath.replace('tracked_persons', 'tracked_skeletons').replace('.jpg', '.npy')
    if not os.path.exists(out_file):
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)

        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        joints = datum.poseKeypoints

        if not args.no_display:
            cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
            cv2.waitKey(0)

        if args.save_skeletons:
            np_joints = np.array(joints).squeeze(axis=0)  # single person
            out_folder = '/'.join(out_file.split('/')[:-1])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            np.save(out_file, np_joints)

