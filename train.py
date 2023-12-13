# Packages
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import datetime
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from matplotlib import pyplot as plt
from detectron2.data.datasets import register_coco_instances


classes=[]
with open('./dataset/train/dataset.json') as f:
    imgs_anns=json.load(f)
    classes=imgs_anns['categories']
print("categories: ")
print(classes)



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

#build trainer
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


# set train config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
cfg.DATASETS.TRAIN = ("dish_train",)
cfg.DATASETS.TEST = ("dish_valid",)
cfg.TEST.EVAL_PERIOD = 500
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "./model_final_f97cb7.pkl"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 500
# Epoch = MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

cfg.OUTPUT_DIR = os.path.join('output', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

# train model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

with open(os.path.join(cfg.OUTPUT_DIR, 'output.yaml'), "w") as f:
    f.write(cfg.dump())

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# evaluate
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("dish_valid", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "dish_valid")
print(inference_on_dataset(predictor.model, val_loader, evaluator))