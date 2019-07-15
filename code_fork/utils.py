import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion

# Download the latest COCO weights

COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

# Bounding Boxes

def extract_bboxes(mask):
    """
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    :param mask:
    :return: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:,:,i]
        # Bounding Box
        horizontalz_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontalz_indicies.shape[0]:
            x1, x2 = horizontalz_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def compute_iou(box, boxes, box_area, boxes_area):
    """
    Calculates IOU of the given box with the array of the given boxes.
    :param box: 1D vector [y1, x1, y2, x2]
    :param boxes: [boxes_count, (y1, x1, y2, x2)]
    :param box_area: float. the area of 'box'
    :param boxes_area: array of length boxes_count.

    Note : the areas are passed in rather than calculated here of
    efficiency. Calculate once in the caller to avoid duplicate work.
    :return:
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    """
    Computes IOU overloaps between two sets of boxes.
    :param boxes1: [N, (y1, x1, y2, x2)].
    :param boxes2:
    :return:
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] -boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] -boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2= boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_overlaps_masks(masks1, masks2):
    """"""
    if masks1.shape[-1] == 0 or masks2.shape[-1] ==0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flattten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    intersections= np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections /union

    return overlaps

def non_max_suppression(boxes, scores, threshold):
    """
    Performs non-maximum suppression and returns indices of kept boxes.
    :param boxes: [N, (y1, x1, y2, x2)]. Notice (y2, x2) lays outside the box.
    :param scores: 1d array of box scores.
    :param threshold:
    :return:
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 -y1) * (x2 - x1)

    #
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])

        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """
    Applies the given deltas to the given boxes.
    :param boxes:
    :param deltas: [N, (dy, dx, log(dh), log(dw))]
    :return:
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:,2] -boxes[:, 0]
    width = boxes[:, 3] -boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    #
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    #
    y1 = center_y -0.5* height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    box = tf.cast(box, tf.float32)  # change type
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:,3] - box[:,1]
    center_y = box[:, 0] + 0.5 *height
    center_x = box[:, 1] + 0.5*width

    gt_height = gt_box[:, 2] -gt_box[:, 0]
    gt_width= gt_box[:, 3] - gt_box[:,1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width
    dy = (gt_center_y - center_y) /height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """
    compute refinement needed to transform box to gt_box
    :param box:
    :param gt_box:
    :return:
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - gt_center_y) / height
    dx = (gt_center_x - gt_center_x) /width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)

# Dataset

class Dataset(object):
    def __index__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id":0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source , class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path , **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """
        Return a link to the image in its source
        :param image_id:
        :return:
        """
        return ""

    def prepare(self, class_map=None):
        """

        :param class_map:not supported yet.
        :return:
        """
        def clean_name(name):
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']):id
                                        for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']):id
                                        for info, id in zip(self.image_info, self.image_ids)}


        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}

        for source in self.sources:
            self.source_class_ids[source] = []
            #
            for i, info in enumerate(self.class_info):
                if i == 0 or source==info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):

        return  self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """

        :param image_id:
        :return:return a [H,W, 3] Numpy array.
        """
        image= skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image= skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
                image = image[..., :3]
        return image

    def load_mask(self, image_id):
        logging.warning("You are using the default load_mask(), maybe you need to define your "
                        "own noe.")
        mask = np.empty([0,0,0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
        image_dtype = image.dtype

        h, w  =image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0),(0,0), (0,0)]
        crop = None

        if mode == "none":
            return image, window, scale, padding, crop
        if min_dim:
            scale = max(1, min_dim / min(h, w))
        if min_scale and scale < min_scale:
            scale = min_scale

        if max_dim and mode == "square":
            h, w = image[:2]
            top_pad = (max_dim - h) //2
            bottom_pad = max_dim - h -top_pad
            left_pad = (max_dim - w) //2
            right_pad = max_dim - w -left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad),(0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0 )
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "pad64":
            h, w = image.shape[:2]
            assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "crop":
            h, w  =image.shape[:2]
            y = random.randint(0, (h - min_dim))
            x = random.randint(0, (9))

