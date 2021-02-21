# Learning Group Activities from Skeletons without Individual Action Labels. [PDF](http://www.micc.unifi.it/seidenari/publication/icpr-2020-1/icpr-2020-1.pdf)

## Requirements
```
Ubuntu >= 16.04
NVIDIA Container Toolkit
Docker Compose
```


## Dataset and Pretrained Network
Download volleyball dataset from author's provided [link](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and extract zip in ./SkeletonGroupActivityRecognition/volleyball_dataset/
Download pretrained-3d-cnn weights from author's provided [link](https://drive.google.com/drive/folders/1u_l-yvhS0shpW6e0tCiqPE7Bd1qQZKdD) and put .tar file in ./SkeletonGroupActivityRecognition/Weights/

## Docker container build
```
git clone https://github.com/fabiozappo/SkeletonGroupActivityRecognition.git 
cd SkeletonGroupActivityRecognition
docker-compose build
```

## Person tracking
```
docker docker-compose run --rm skeleton-group-activity-recognition exec python VDtracker.py
```

## Skeleton extraction 
```
docker docker-compose run --rm openpose exec python extract_skeletons.py --no_display --save
```


## Group Activity Recognition
```
docker docker-compose run --rm skeleton-group-activity-recognition exec python train.py --augment --pivot --pseudo_labels
```
