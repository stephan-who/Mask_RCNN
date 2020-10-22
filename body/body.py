"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import random
import numpy as np
import skimage.draw
import skimage.io
import yaml
import cv2
import matplotlib.pyplot as plt
 
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from PIL import Image
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

IMAGE_DIR = os.path.join(ROOT_DIR, "images")

item_num = 0

############################################################
#  Configurations
############################################################


class BodyConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "body"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    TRAIN_ROIS_PER_IMAGE = 32

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.9
    VALIDATION_STEPS = 5

############################################################
#  Dataset
############################################################

class BodyDataset(utils.Dataset):
    # 得到该图中有多少个示例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # Rewrite
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index+1:
                        mask[j, i, index] = 1
        return mask


    # Rewrite load_shapes, 里面包含自己的类别，可以任意添加
    def load_bodies(self, count, img_folder, mask_folder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
                count: number of images to generate.
                height, width: the size of the generated images.
                """
        # Add body classes
        self.add_class("shapes", 1, "")
        self.add_class("shapes", 2, "")
        self.add_class("shapes", 3, "")
        for i in range(count):
            file_str = imglist[i].split(".")[0]
            mask_path = mask_folder + '/' + file_str + '.png'
            yaml_path = dataset_root_path + '/labelme_json' + file_str + '_json/info.yaml'
            print(dataset_root_path + '/labelme_json' + file_str + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + '/labelme_json/' + file_str + "_json/img.png")
            self.add_image("shapes", image_id=i, path=img_folder + '/' +imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # Rewrite load_mask
    def load_mask(self, image_id):
        """Generate instance masks for the given image ID.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        global iter_num
        info = self.image_info[image_id]
        count = 1 # number of boject
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("") != -1:
                labels_form.append("")


        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "body":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def get_ax(rows=1, cols=1, size=8):

    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def train_model():
    """Train the model."""
    # Training dataset.
    dataset_root_path = r"train_data/"
    img_folder = os.path.join(dataset_root_path, 'pic')
    mask_folder = os.path.join(dataset_root_path, 'cv2_mask')

    img_list = os.listdir(img_folder)
    count = len(img_list)

    dataset_train = BodyDataset()
    dataset_train.load_bodies(count, img_folder, mask_folder, img_list, dataset_root_path)
    dataset_train.prepare()

    dataset_val = BodyDataset()
    dataset_val.load_bodies(7, img_folder, mask_folder, img_list, dataset_root_path)
    dataset_val.prepare()

    config = BodyConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # 第一次训练时，这里填coco，在产生训练后的模型后，改成last
    init_with = "last"  # imagenet, coco or last
    if init_with  == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                                                   "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # load last models you trained and continue training
        checkpoint_file = model.find_last()
        model.load_weights(checkpoint_file, by_name=True)

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=30,
                layers="all")

class InferenceConfig(BodyConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def predict():

    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    model_path = model.find_last()

    # Load trained weights
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from", model_path)
    model.load_weights(model_path, by_name=True)

    class_names = ['BG', '', '']

    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    
    results = model.detect([image], verbose=1)
    
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

if __name__ == '__main__':
    train_model()
    
    import argparse

    