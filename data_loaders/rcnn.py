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

    def __init__(self, df, test=False, transforms=None, write_path=None, print_augmentations=False):
        super(WheatDataset, self).__init__()
        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = Config.train_images_dir
        self.transforms = transforms
        self.write_path = write_path
        self.print_augmentations = print_augmentations
        self.test = test

    # noinspection PyUnboundLocalVariable
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        load_image_type = 0
        if not self.test:
            load_image_type = random.choice(range(2))
            if load_image_type == 0:
                image, boxes = self.load_image_and_boxes(index)
            elif load_image_type == 1:
                image, boxes = self.load_cutmix_image_and_boxes(index)
        else:
            image, boxes = self.load_image_and_boxes(index)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        is_crowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {'boxes': boxes,
                  'labels': labels,
                  'image_id': torch.tensor([index]),
                  'area': area,
                  'iscrowd': is_crowd}

        if load_image_type != 0:
            self.transforms = get_valid_transforms()

        if self.transforms:
            for i in range(10):
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transforms(**sample)
                if len(sample['bboxes']) > 0:
                    if self.print_augmentations:
                        print(image_id)
                        print(sample['replay'])
                        print()
                    image = sample['image']
                    target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
                    target['labels'] = torch.ones((target['boxes'].shape[0],), dtype=torch.int64)
                    break

        if self.write_path:
            write_img_to_disk(image.permute(1, 2, 0).cpu().numpy(),
                              target['boxes'].cpu().numpy().tolist(),
                              file_path=self.write_path / f'{image_id}.jpg')
        return image, target, image_id

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

    def load_cutmix_image_and_boxes(self, index, img_size=1024):
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


if __name__ == '__main__':
    data = get_data()
    dataset = iter(WheatDataset(data, test=False, transforms=get_train_transforms(),
                                write_path=Path.cwd(), print_augmentations=False))
    for _ in range(10):
        next(dataset)
