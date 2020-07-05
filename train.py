import random
import warnings
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer, callbacks
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import Config
from data_loaders import rcnn
from modules.rcnn import FasterRCNNResnet50FPN
from utils.augmentations import get_valid_transforms, get_train_transforms
from utils.data import get_data, collate_fn


def train():
    data_df = get_data()
    train_ids, oof_ids = train_test_split(data_df['image_id'].unique(), test_size=0.10,
                                          shuffle=True, random_state=Config.seed)
    train_df = data_df.loc[data_df['image_id'].isin(train_ids)]
    oof_df = data_df.loc[data_df['image_id'].isin(oof_ids)]

    train_dataset = rcnn.WheatDataset(train_df, transforms=get_train_transforms())
    train_dataloader = DataLoader(train_dataset, batch_size=Config.Train.batch_size,
                                  shuffle=True, num_workers=4, drop_last=True,
                                  collate_fn=collate_fn, pin_memory=True)
    oof_dataset = rcnn.WheatDataset(oof_df, test=True, transforms=get_valid_transforms())
    oof_dataloader = DataLoader(oof_dataset, batch_size=Config.Train.batch_size,
                                shuffle=False, num_workers=4,
                                collate_fn=collate_fn, pin_memory=True)
    model = FasterRCNNResnet50FPN.load_from_checkpoint('checkpoints\\faster_rcnn\\epoch=10.ckpt', **Config)
    # model = FasterRCNNResnet50FPN(conf=Config)
    early_stop = callbacks.EarlyStopping(monitor='val_loss',
                                         patience=5,
                                         mode='min',
                                         verbose=True)
    checkpoint = callbacks.ModelCheckpoint(str(Config.Train.checkpoint_dir),
                                           monitor='val_loss',
                                           verbose=True,
                                           save_top_k=1)
    cbs = [
        callbacks.LearningRateLogger()
    ]
    trainer = Trainer(gpus=1,
                      early_stop_callback=early_stop,
                      checkpoint_callback=checkpoint,
                      callbacks=cbs,
                      benchmark=True,
                      deterministic=True,
                      max_epochs=Config.Train.epochs)
    trainer.fit(model, train_dataloader=train_dataloader,
                val_dataloaders=oof_dataloader)

    valid_dataset = rcnn.WheatDataset(get_data(mode='valid'), test=True, transforms=get_valid_transforms())
    valid_dataloader = DataLoader(valid_dataset, batch_size=Config.Train.batch_size,
                                  shuffle=False, num_workers=4,
                                  collate_fn=collate_fn, pin_memory=True)
    trainer.test(model, test_dataloaders=valid_dataloader)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(Config.seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = True
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.enabled = True

    Path(Config.Train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train()
