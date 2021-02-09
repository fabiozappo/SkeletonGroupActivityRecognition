import argparse
import time
import cv2
import pyopenpose as op

# os.system('./build/examples/openpose/openpose.bin --image_dir ' + frame_folder_path + ' --write_json ' + save_json_path + ' --number_people_max 1 --tracking 0 --display 0 --render_pose 0 --num_gpu_start 6 --num_gpu 1')


# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="/work/code/Datasets/tracked_persons/",
                    help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--output_dir", default="/work/code/Datasets/extracted_skeletons/",
                    help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", action='store_true', help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/openpose/models/"
params["number_people_max"] = 1

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
        if key not in params: params[key] = next_item

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Read frames on directory
imagePaths = op.get_images_on_directory(args[0].image_dir);
start = time.time()

# Process and display images
for imagePath in imagePaths:
    datum = op.Datum()
    imageToProcess = cv2.imread(imagePath)

    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    joints = datum.poseKeypoints

    save_json()

    print("Body keypoints: \n" + str(joints))

    if not args[0].no_display:
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(0)

end = time.time()
print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

