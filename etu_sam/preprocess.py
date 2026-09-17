"""Image / box preprocessing matching ETU-SAM evaluation (256², ImageNet z-score)."""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch


class DataPreprocessor:
    def __init__(self, image_size: int = 256, bbox_shift: int = 10):
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def preprocess(self, image, mask, *, box_jitter: bool = False, box_expand: int | None = None, sample_instance: bool = False):
        img_3c = image
        img_resize = self.resize_longest_side(img_3c)
        img_resize = (img_resize.astype(np.float32) - self.pixel_mean) / self.pixel_std
        img_padded = self.pad_image(img_resize)
        img_padded = np.transpose(img_padded, (2, 0, 1))

        gt = mask
        gt = cv2.resize(gt, (img_resize.shape[1], img_resize.shape[0]), interpolation=cv2.INTER_NEAREST).astype(
            np.uint8
        )
        gt = self.pad_image(gt)
        if sample_instance:
            label_ids = np.unique(gt)[1:]
            gt2d = np.uint8(gt == (random.choice(label_ids.tolist()) if label_ids.size else np.max(gt)))
        else:
            gt2d = np.uint8(gt > 0)
        y_indices, x_indices = np.where(gt2d > 0)
        if y_indices.size == 0:
            bboxes = np.array([0, 0, self.image_size - 1, self.image_size - 1], dtype=np.float32)
        else:
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            h, w = gt2d.shape
            shift = int(box_expand) if box_expand is not None else self.bbox_shift
            if box_jitter:
                x_min = max(0, x_min - random.randint(0, shift))
                x_max = min(w - 1, x_max + random.randint(0, shift))
                y_min = max(0, y_min - random.randint(0, shift))
                y_max = min(h - 1, y_max + random.randint(0, shift))
            else:
                x_min = max(0, x_min - shift)
                x_max = min(w - 1, x_max + shift)
                y_min = max(0, y_min - shift)
                y_max = min(h - 1, y_max + shift)
            bboxes = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2d[None, :, :]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(),
            "resized_hw": (int(img_resize.shape[0]), int(img_resize.shape[1])),
        }

    def encode_image(self, image):
        img_resize = self.resize_longest_side(image)
        resized_hw = (int(img_resize.shape[0]), int(img_resize.shape[1]))
        img_n = (img_resize.astype(np.float32) - self.pixel_mean) / self.pixel_std
        img_padded = self.pad_image(img_n)
        tensor = torch.tensor(np.transpose(img_padded, (2, 0, 1))).float()
        return tensor, resized_hw

    def transform_box(self, box_xyxy, orig_hw, expand: int = 15):
        oldh, oldw = orig_hw
        newh, neww = self.letterbox_hw(orig_hw)
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        x1 = x1 * neww / max(oldw, 1)
        x2 = x2 * neww / max(oldw, 1)
        y1 = y1 * newh / max(oldh, 1)
        y2 = y2 * newh / max(oldh, 1)
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(self.image_size - 1, x2 + expand)
        y2 = min(self.image_size - 1, y2 + expand)
        return torch.tensor([[[[x1, y1, x2, y2]]]]).float()

    def letterbox_hw(self, orig_hw):
        oldh, oldw = orig_hw
        scale = self.target_length * 1.0 / max(oldh, oldw)
        return int(oldh * scale + 0.5), int(oldw * scale + 0.5)

    def restore(self, arr, orig_hw, nearest: bool = False):
        newh, neww = self.letterbox_hw(orig_hw)
        crop = arr[:newh, :neww]
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(crop, (orig_hw[1], orig_hw[0]), interpolation=interp)

    def resize_longest_side(self, image):
        oldh, oldw = image.shape[0], image.shape[1]
        scale = self.target_length * 1.0 / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
        return cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if image.ndim == 3:
            return np.pad(image, ((0, padh), (0, padw), (0, 0)))
        return np.pad(image, ((0, padh), (0, padw)))
