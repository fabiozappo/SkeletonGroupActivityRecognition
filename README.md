# Learning Group Activities from Skeletons without Individual Action Labels. http://www.micc.unifi.it/seidenari/publication/icpr-2020-1/icpr-2020-1.pdf

## Requirements
Ubuntu > 16.04
docker-nvidia
docker-compose
nvidia-driver
Download volleyball dataset in ./SkeletonGroupActivityRecognition/volleyball_dataset/
Download pretrained-3d-cnn wehighrs in ./SkeletonGroupActivityRecognition/Weights/

## Docker container build
git clone ** 
cd SkeletonGroupActivityRecognition
docker-compose build

## Person tracking
docker docker-compose run --rm skeleton-group-activity-recognition exec python VDtracker.py

## Skeleton extraction 
docker docker-compose run --rm openpose exec python extract_skeletons.py --no_display --save

## Group Activity Recognition
docker docker-compose run --rm skeleton-group-activity-recognition exec python train.py --augment --pivot --pseudo_labels
