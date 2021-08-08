# Learning Group Activities from Skeletons without Individual Action Labels.
#### This is an official implementation of Learning Group Activities from Skeletons without Individual Action Labels paper [[Link]](http://www.micc.unifi.it/seidenari/publication/icpr-2020-1/icpr-2020-1.pdf).

## Requirements
```
Ubuntu >= 16.04
NVIDIA Container Toolkit
Docker Compose
```
For instructions on getting started with the NVIDIA Container Toolkit, refer to the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

## Dataset and Pretrained Network
- Download volleyball dataset from author's provided [link](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and extract zip in ./SkeletonGroupActivityRecognition/volleyball_dataset/ 
- Download pretrained-3d-cnn weights from author's provided [link](https://drive.google.com/drive/folders/1u_l-yvhS0shpW6e0tCiqPE7Bd1qQZKdD) and put .tar file in ./SkeletonGroupActivityRecognition/Weights/

## Docker container build
```
git clone https://github.com/fabiozappo/SkeletonGroupActivityRecognition.git 
cd SkeletonGroupActivityRecognition
docker-compose build
```

## Person tracking
To enter in the container and track person in videos use:
```
docker-compose run --rm skeleton-group-activity-recognition
python VDtracker.py
```
Ctrl + D to exit from the container

## Skeleton extraction 
To enter in the openpose container and extract skeletons from each person tracklet:
```
docker-compose run --rm openpose 
python3 extract_skeletons.py --no_display --save
```
Ctrl + D to exit from the container


## Group Activity Recognition
```
docker docker-compose run --rm skeleton-group-activity-recognition 
python train.py --augment --pivot --pseudo_labels
```
Ctrl + D to exit from the container

## Citation
If you find this code to be useful in your own research, please consider citing our paper:
```bib
@inproceedings{Zappardino2020LearningGA,
  title={Learning Group Activities from Skeletons without Individual Action Labels},
  author={Fabio Zappardino and Tiberio Uricchio and Lorenzo Seidenari and A. D. Bimbo},
  year={2020}
}
```
