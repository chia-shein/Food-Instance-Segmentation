import os, random, cv2, json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from utils import imshow
classes=[]
with open('./dataset/train/dataset.json') as f:
    imgs_anns=json.load(f)
    classes=imgs_anns['categories']

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dish_metadata = MetadataCatalog.get("dish_train")

test_dir = './dataset/test/img'

for file_name in random.sample(os.listdir(test_dir), 6):
    test_file_path = os.path.join(test_dir, file_name)
    im = cv2.imread(test_file_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
            metadata=dish_metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    imshow(out.get_image())