import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Config


def get_data(mode='train'):
    if mode == 'train':
        df = pd.read_csv(Config.train_csv)
    elif mode == 'valid':
        df = pd.read_csv(Config.valid_csv)
    b_boxes = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = b_boxes[:, i]
    df.drop(columns=['bbox'], inplace=True)
    df['source'] = df.source.astype('category')
    return df


def collate_fn(batch):
    return tuple(zip(*batch))


def visualize_bbox(img, bbox, bb_format='pascal_voc'):
    x_min, y_min = bbox[:2]
    if bb_format == 'coco':
        w, h = bbox[-2:]
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    else:
        x_min, x_max, y_min, y_max = int(x_min), int(bbox[2]), int(y_min), int(bbox[3])
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
    # ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    # cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    # cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR,
    #             lineType=cv2.LINE_AA)
    return img


def write_img_to_disk(img, bboxes, return_image=False, file_path=None, heatmap=None, bb_format='pascal_voc'):
    image = img.copy()
    for bbox in bboxes:
        image = visualize_bbox(image, bbox, bb_format=bb_format)
    plt.figure(figsize=(img.shape[0]/120, img.shape[1]/120), dpi=120)
    plt.imshow(image)
    if heatmap is not None:
        plt.imshow(255 * heatmap, alpha=0.5, cmap='viridis')
    plt.axis('off')
    if return_image:
        return image
    else:
        # cv2.imwrite(str(file_path), (image * 255).astype('int'))
        plt.savefig(str(file_path), bbox_inches='tight', pad_inches=0)
    plt.close()


def get_image_with_results(img, preds, gt_boxes):
    for pred_box in preds:
        cv2.rectangle(
            img,
            (pred_box[0], pred_box[1]),
            (pred_box[2], pred_box[3]),
            (1, 0, 0), 2
        )

    for gt_box in gt_boxes:
        cv2.rectangle(
            img,
            (gt_box[0], gt_box[1]),
            (gt_box[2], gt_box[3]),
            (0, 0, 1), 2
        )
    return img

