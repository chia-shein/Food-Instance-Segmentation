'''
    This code is for checking the Label data
    show the image with its label.
'''

import os, random, cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from utils import imshow

train_ann_path = r'./dataset/train/dataset.json'
train_img_path = r'./dataset/train/img'
valid_ann_path = r'./dataset/valid/dataset.json'
valid_img_path = r'./dataset/valid/img'

assert os.path.isfile(train_ann_path), f'Training annotation file not found: {train_ann_path}'
assert os.path.isdir(train_img_path), f'Training image directory not found: {train_img_path}'
assert os.path.isfile(valid_ann_path), f'Validation annotation file not found: {valid_ann_path}'
assert os.path.isdir(valid_img_path), f'Validation image directory not found: {valid_img_path}'

register_coco_instances("dish_train", {}, train_ann_path, train_img_path)
register_coco_instances("dish_valid", {}, valid_ann_path, valid_img_path)


dataset_dicts = DatasetCatalog.get("dish_train")
dish_metadata = MetadataCatalog.get("dish_train")

#random sample three images and its label.
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=dish_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    imshow(out.get_image())