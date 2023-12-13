# Food-Instance-Segmentation
## NCKU "Introduction to neural network" - (**Final Project**)
![](./readme_images/dataset.png)

### Categories:
1. staple food
2. main course
3. side dish
4. vegetable

### Dataset Description:
* [Download Link](https://drive.google.com/drive/folders/1uu-1P_gHhqjskyVrUhJIfH9uUvaInEiT?usp=sharing)

|  | staple food | main course | side dish | vegetable | total |
| :--: | :--: | :--: | :--: | :--: | :--:|
| instances | 1174 | 1022 | 965 | 1268 | 4429 |

### Label Tool: 
1. Detection task label tool: [LabelImg](https://github.com/HumanSignal/labelImg)
2. Segmetation task label tool: [Labelme](https://github.com/wkentaro/labelme)

### Dependencies
* pytorch-21.06-py3:latest

* Download the pretrained weight: [Download link](https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl)
   ```shell
      wget -q https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl 
* Install others related library and dependencies:
   ```shell
      sudo apt-get update
      sudo apt-get install ffmpeg libsm6 libxext6  -y
   ```
### Code
#### show_images.py
* Check the images and the annotation inside the dataset.
```shell
python show_images.py
```
![](./readme_images/showimg.png)

#### train.py
* training the model also evaluate the trained model.
```shell
python train.py
```

#### inference.py
* show the test images predicted by the trained model.
```shell
python inference.py
```

### Experiment Results

