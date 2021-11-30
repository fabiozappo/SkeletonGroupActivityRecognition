# Learning Group Activities from Skeletons without Individual Action Labels.
#### This is an official implementation of Learning Group Activities from Skeletons without Individual Action Labels paper [[Link]](http://www.micc.unifi.it/seidenari/publication/icpr-2020-1/icpr-2020-1.pdf).

## Requirements
```
Ubuntu >= 16.04
NVIDIA Container Toolkit
```
For instructions on getting started with the NVIDIA Container Toolkit, refer to the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

## Dataset and Pretrained Network
- Download volleyball dataset from author's provided [link](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and extract zip in ./SkeletonGroupActivityRecognition/volleyball_dataset/ 
- (In case use don't use the Docker container) Download pretrained-3d-cnn weights from author's provided [link](https://drive.google.com/drive/folders/1u_l-yvhS0shpW6e0tCiqPE7Bd1qQZKdD) and put .tar file in ./SkeletonGroupActivityRecognition/Weights/

```
SkeletonGroupActivityRecognition
|-- ...
|-- Dockerfile
|-- VDtracker.py
|-- Weights
|   |-- p3d_flow_199.checkpoint.pth.tar
|   `-- p3d_rgb_199.checkpoint.pth.tar
|-- extract_skeletons.py
|-- train.py
`-- volleyball_dataset
    `-- videos
```

## Docker container
Build the container:
cd into cloned repo and run:
```
docker build -t skeleton-group-activity-recognition:latest .
```
Run the container:
```
docker run \
  --rm -it \
  --gpus="device=all" \
  -v volleyball_dataset:/work/sk-gar/volleyball_dataset \
  skeleton-group-activity-recognition
```

- `-it` & `--rm`: Starts the container with an interactive session and removes the container after closing the session.
- optional: `--gpus='"device=0,1,2,3"'` restrict the GPU devices the container can access.
- `-v ./volleyball_dataset:/work/sk-gar/volleyball_dataset`: Makes the host's folder `./volleyball_dataset` available inside the container at `/work/sk-gar/volleyball_dataset`
- `skeleton-group-activity-recognition` the name/tag of the image

## Running the scripts

### Person tracking & Skeleton extraction 
```
python3 VDtracker.py
python3 extract_skeletons.py --no_display --save
```
After running, there should be two directories `tracked_persons` and `tracked_skeletons`.
```
SkeletonGroupActivityRecognition
`-- volleyball_dataset
    |-- tracked_persons
    |-- tracked_skeletons
    `-- videos
```
### Group Activity Recognition

- Group activity labels only with data augmentation
  ```
  python train.py --augment --pivot --loss_balancer 0
  ```
- Group activity labels without data augmentation
  ```
  python train.py --pivot --loss_balancer 0
  ```
- Pseudo action labels from 3D-Resnet with augmentation
  ```
  python train.py --augment --pivot --pseudo_labels --loss_balancer 0
  ```
- Pseudo action labels from 3D-Resnet without augmentation
  ```
  python train.py --pivot --pseudo_labels --loss_balancer 0
  ```
- Supervised with data augmentation
  ```
  python train.py --augment --pivot
  ```
- Supervised without data augmentation
  ```
  python train.py --pivot
  ```

## Citation
If you find this code to be useful in your own research, please consider citing our paper:
```bib
@inproceedings{zappardinoLearningGroupActivities2021,
  title = {Learning {{Group Activities}} from {{Skeletons}} without {{Individual Action Labels}}},
  booktitle = {2020 25th {{International Conference}} on {{Pattern Recognition}} ({{ICPR}})},
  author = {Zappardino, Fabio and Uricchio, Tiberio and Seidenari, Lorenzo and del Bimbo, Alberto},
  date = {2021-01},
  pages = {10412--10417},
  issn = {1051-4651},
  doi = {10.1109/ICPR48806.2021.9413195},
  abstract = {To understand human behavior we must not just recognize individual actions but model possibly complex group activity and interactions. Hierarchical models obtain the best results in group activity recognition but require fine grained individual action annotations at the actor level. In this paper we show that using only skeletal data we can train a state-of-the art end-to-end system using only group activity labels at the sequence level. Our experiments show that models trained without individual action supervision perform poorly. On the other hand we show that pseudo-labels can be computed from any pre-trained feature extractor with comparable final performance. Finally our carefully designed lean pose only architecture shows highly competitive results versus more complex multimodal approaches even in the self-supervised variant.},
  eventtitle = {2020 25th {{International Conference}} on {{Pattern Recognition}} ({{ICPR}})},
  keywords = {Activity recognition,Annotations,Art,Computational modeling,Computer architecture,Data privacy,Feature extraction}
}
```
