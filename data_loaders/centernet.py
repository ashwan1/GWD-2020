import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config
from utils.augmentations import get_valid_transforms, get_train_transforms
from utils.data import write_img_to_disk, get_data


class WheatDataset(Dataset):
    def __init__(self, df, transforms, min_overlap=0.7, k=1, test=False, write_path=None):
        super(WheatDataset, self).__init__()
        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = Config.train_images_dir
        self.transforms = transforms
        self.min_overlap = min_overlap
        self.k = k
        self.test = test
        self.write_path = write_path
        self.img_size = Config.Train.img_size

    def __getitem__(self, index):
        image = None

        bboxes = None
        # Only 1 class of objects
        center_heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        width_heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        height_heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        # heatmap, width, height
        target = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)

        load_image_type = 0
        if not self.test:
            load_image_type = random.choice(range(2))
            if load_image_type == 0:
                image, bboxes = self.load_image_and_boxes(index)
            elif load_image_type == 1:
                image, bboxes = self.load_mosaic_image_and_boxes(index)
        else:
            image, bboxes = self.load_image_and_boxes(index)

        if load_image_type != 0:
            self.transforms = get_valid_transforms()

        if self.transforms:
            for i in range(10):
                sample = {
                    'image': image,
                    'bboxes': bboxes,
                    'labels': np.ones((bboxes.shape[0],))
                }
                sample = self.transforms(**sample)
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    bboxes = sample['bboxes']
                    break

        for bbox in bboxes:
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = int(self.gaussian_radius((math.ceil(h), math.ceil(w))))
                center = np.array([(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2], dtype=np.int32)
                self.draw_gaussian(center_heatmap, width_heatmap, height_heatmap,
                                   w, h, center, radius)

        if self.write_path:
            write_img_to_disk(image.permute(1, 2, 0).cpu().numpy(),
                              bboxes,
                              file_path=self.write_path / f'{self.image_ids[index]}.jpg',
                              heatmap=width_heatmap)
        target[0] = center_heatmap
        target[1] = height_heatmap
        target[2] = width_heatmap
        return image, torch.tensor(target)

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_mosaic_image_and_boxes(self, index, img_size=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = img_size, img_size
        s = img_size // 2

        xc, yc = [int(random.uniform(img_size * 0.25, img_size * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((img_size, img_size, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            # noinspection PyUnboundLocalVariable
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes

    def gaussian_radius(self, box_dimensions):
        """Adapted from https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py#L95"""
        height, width = box_dimensions

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - self.min_overlap) / (1 + self.min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - self.min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * self.min_overlap
        b3 = -2 * self.min_overlap * (height + width)
        c3 = (self.min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return max(0, min(r1, r2, r3))

    def draw_gaussian(self, c_heatmap, w_heatmap, h_heatmap, w, h, center, radius):
        """Adapted from https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py#L126"""
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = c_heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_c_heatmap = c_heatmap[y - top:y + bottom, x - left:x + right]
        masked_w_heatmap = w_heatmap[y - top:y + bottom, x - left:x + right]
        masked_h_heatmap = h_heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_c_heatmap.shape) > 0:
            np.maximum(masked_c_heatmap, masked_gaussian * self.k, out=masked_c_heatmap)

        masked_w_heatmap_new = (w / self.img_size) * masked_gaussian
        if min(masked_gaussian.shape) > 0 and min(masked_w_heatmap.shape) > 0:
            np.maximum(masked_w_heatmap, masked_w_heatmap_new, out=masked_w_heatmap)

        masked_h_heatmap_new = (h / self.img_size) * masked_gaussian
        if min(masked_gaussian.shape) > 0 and min(masked_h_heatmap.shape) > 0:
            np.maximum(masked_h_heatmap, masked_h_heatmap_new, out=masked_h_heatmap)

        return masked_c_heatmap, masked_w_heatmap, masked_h_heatmap

    # noinspection PyMethodMayBeStatic
    def gaussian_2d(self, shape, sigma):
        """Adapted from https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py#L118"""
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h


if __name__ == '__main__':
    data = get_data()
    wp = Path.cwd() / 'test_images'
    wp.mkdir(parents=True, exist_ok=True)
    dataset = iter(WheatDataset(data, get_train_transforms(), test=False, min_overlap=0.7,
                                write_path=wp))
    for _ in range(20):
        next(dataset)
