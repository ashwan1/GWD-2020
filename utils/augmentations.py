import albumentations as A
from albumentations import BboxParams
from albumentations.pytorch import ToTensorV2

from config import Config


def get_train_transforms():
    # noinspection PyTypeChecker
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(850, 850), height=1024, width=1024, p=0.3),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                 val_shift_limit=0.2, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.9)
        ], p=0.5),
        A.OneOf([
            A.RandomRain(rain_type='drizzle', p=0.2),
            A.GaussianBlur(blur_limit=7, p=0.5),
            A.GaussNoise((0.2, 0.3), p=0.3),
            A.RandomShadow(p=0.2)
        ], p=0.4),
        A.ToGray(p=0.01),
        A.Flip(p=0.5),
        A.CoarseDropout(max_height=64, max_width=64, min_holes=3,
                        min_height=32, min_width=32, p=0.5),
        A.Resize(Config.Train.img_size, Config.Train.img_size, p=1.0),
        A.Normalize(max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params=BboxParams('pascal_voc', label_fields=['labels'], min_visibility=0.2))


def get_valid_transforms():
    return A.Compose([
        A.Resize(Config.Train.img_size, Config.Train.img_size, p=1.0),
        A.Normalize(max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params=BboxParams('pascal_voc', label_fields=['labels'], min_visibility=0.2))


def get_test_transforms():
    return A.Compose([
        A.Resize(Config.Train.img_size, Config.Train.img_size, p=1.0),
        A.Normalize(max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params=BboxParams('pascal_voc', label_fields=['labels']))
