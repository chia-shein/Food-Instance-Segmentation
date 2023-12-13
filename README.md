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

| category | staple food | main course | side dish | vegetable | total |
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
1. Evaluation results for bbox:

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 61.920 | 80.313 | 74.643 | 12.612 | 47.781 | 68.195 |

2. Per-category bbox AP:

| category | staple food | main course | side dish | vegetable |
| :------: | :------: | :------: | :------: | :------: |
| AP | 78.647 | 52.682 | 48.953 | 67.399 |

3. Evaluation results for segm:

|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 44.442 | 78.253 | 49.255 | 2.878 | 27.602 | 51.990 |

4. Per-category segm AP:

| category | staple food | main course | side dish | vegetable |
| :------: | :------: | :------: | :------: | :------: |
| AP | 57.265 | 34.415 | 32.760 | 53.329 |

5. Prediction Result:
