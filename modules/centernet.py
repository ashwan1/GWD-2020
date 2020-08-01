import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR, ReduceLROnPlateau

from custom.lr_schedulers.n_cycle import NCycleLR
from custom.optimizers import RangerLars
from modules.helper_mods import ASPP, ConvBnRelu
from utils.data import get_image_with_results, write_img_to_disk, visualize_bbox
from utils.losses_n_metrics import calculate_image_precision


# noinspection PyAbstractClass
class Resnest50CenterNet(LightningModule):
    def __init__(self, conf=None, *args, **kwargs):
        super(Resnest50CenterNet, self).__init__()
        self.hparams = conf
        resnest = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

        self.conv1 = resnest.conv1  # 64x512x512
        self.bn1 = resnest.bn1
        self.relu = resnest.relu
        self.maxpool = resnest.maxpool  # 64x256x256
        self.layer1 = resnest.layer1  # 256x256x256
        self.layer2 = resnest.layer2  # 512x128x128
        self.layer3 = resnest.layer3  # 1024x64x64
        self.aspp = ASPP(1024, 128)
        self.conv2 = ConvBnRelu(256, 128, 1)
        self.conv3 = ConvBnRelu(256, 64, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 1, 1)
        self.conv5 = torch.nn.Conv2d(64, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        y = self.layer1(x)
        x = self.layer2(y)
        x = self.layer3(x)
        x = self.aspp(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        y = self.conv2(y)
        x = torch.cat((x, y), dim=1)
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x_center = self.conv4(x)
        x_center = F.sigmoid(x_center)
        x_wh = self.conv5(x)
        return x_center, x_wh

    def configure_optimizers(self):
        scheduler = None
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = RangerLars(params)
        # noinspection PyUnresolvedReferences
        if self.hparams.Train.scheduler == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer, max_lr=self.hparams.Train.lr,
                                   epochs=self.hparams.Train.epochs,
                                   steps_per_epoch=self.hparams.Train.steps_per_epoch,
                                   pct_start=self.hparams.Train.Schedulers.OneCycleLR.pct_start,
                                   anneal_strategy=self.hparams.Train.Schedulers.OneCycleLR.anneal_strategy,
                                   cycle_momentum=False,
                                   div_factor=self.hparams.Train.Schedulers.OneCycleLR.div_factor)
        elif self.hparams.Train.scheduler == 'NCycleLR':
            scheduler = NCycleLR(optimizer, max_lr=self.hparams.Train.lr,
                                 n=self.hparams.Train.Schedulers.NCycleLR.n,
                                 lr_factor=self.hparams.Train.Schedulers.NCycleLR.lr_factor,
                                 epochs=self.hparams.Train.epochs,
                                 steps_per_cycle=self.hparams.Train.Schedulers.NCycleLR.steps_per_cycle,
                                 pct_start=self.hparams.Train.Schedulers.NCycleLR.pct_start,
                                 anneal_strategy=self.hparams.Train.Schedulers.NCycleLR.anneal_strategy,
                                 cycle_momentum=False,
                                 div_factor=self.hparams.Train.Schedulers.NCycleLR.div_factor)
        elif self.hparams.Train.scheduler == 'CyclicLR':
            scheduler = CyclicLR(optimizer,
                                 base_lr=self.hparams.Train.lr / 1e5,
                                 max_lr=self.hparams.Train.lr,
                                 step_size_up=self.hparams.Train.steps_per_epoch,
                                 mode=self.hparams.Train.Schedulers.CyclicLR.mode,
                                 gamma=self.hparams.Train.Schedulers.CyclicLR.gamma,
                                 cycle_momentum=False)
        elif self.hparams.Train.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer,
                                          factor=self.hparams.Train.Schedulers.ReduceLROnPlateau.factor,
                                          patience=self.hparams.Train.Schedulers.ReduceLROnPlateau.patience,
                                          verbose=True)
        schedulers = [
            {
                'scheduler': scheduler,
                'interval': self.hparams.Train.Schedulers.interval
            }
        ]
        return [optimizer], schedulers

    # noinspection PyMethodMayBeStatic
    def get_losses(self, outputs, targets):
        center_hm = outputs[0]
        hw_hm = torch.clamp(outputs[1], 0, 1)
        center_hm_loss = F.binary_cross_entropy(center_hm, targets[:, 0].view(center_hm.shape))
        # center_hm_loss = self.neg_loss(center_hm, targets[:, 0].view(center_hm.shape))
        hw_hm_loss = F.l1_loss(hw_hm, targets[:, -2:])
        return center_hm_loss, hw_hm_loss

    def neg_loss(self, pred, gt):
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            pred (batch x c x h x w)
            gt (batch x c x h x w)
        """
        pred = pred.unsqueeze(1).float()
        gt = gt.unsqueeze(1).float()

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds
        neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 3) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        center_hm_loss, wh_hm_loss = self.get_losses(outputs, targets)
        loss = center_hm_loss + wh_hm_loss
        lr = self.trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']
        loss_dict = {
            'chm_loss': center_hm_loss,
            'whhm_loss': wh_hm_loss
        }
        return {'loss': loss, 'log': loss_dict, 'progress_bar': {'lr': lr}}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        center_hm_loss, wh_hm_loss = self.get_losses(outputs, targets)
        loss = center_hm_loss + wh_hm_loss
        precisions = np.zeros((self.hparams.Train.batch_size,))
        c_hm = self.nms(outputs[0])
        for i, target in enumerate(targets):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            gt_bboxes = self.get_gt_bboxes(target)
            # write_img_to_disk(img, gt_bboxes, file_path=Path.cwd() / f'gt_{batch_idx}_{i}.jpg')
            pred_bboxes = self.get_pred_bboxes(c_hm[i], outputs[1][i])
            # write_img_to_disk(img, pred_bboxes, file_path=Path.cwd() / f'pred_{batch_idx}_{i}.jpg')
            if pred_bboxes.size == 0:
                precision = 0
            else:
                precision = calculate_image_precision(gt_bboxes, pred_bboxes)
            precisions[i] = precision
        return {'val_loss': loss,
                'val_map': torch.tensor(precisions, dtype=self.dtype, device=self.device)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_map'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_map': avg_precision}
        return {'avg_val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        if batch_idx < 0:
            images, targets = batch
            outputs = self.forward(images)
            c_hm = self.nms(outputs[0])
            for i, output in enumerate(outputs):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                boxes = self.get_pred_bboxes(c_hm[i], outputs[1][i])
                gt = self.get_gt_bboxes(targets[i])
                results = get_image_with_results(image, boxes, gt)
                self.logger.experiment.add_image(f"bb_test(RED: Predicted; BLUE: Ground-truth)/image{i}",
                                                 results, dataformats='HWC')
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        test_results = self.validation_epoch_end(outputs)
        test_results = {k.replace('val', 'test'): v.cpu().numpy().tolist() for k, v in test_results['log'].items()}
        return {'log': test_results}

    # noinspection PyMethodMayBeStatic
    def get_gt_bboxes(self, target):
        img_size = self.hparams.Train.img_size
        h_hm = (target[1] * img_size).cpu().numpy()
        w_hm = (target[2] * img_size).cpu().numpy()
        centers = torch.nonzero(target[0] == 1).cpu().numpy().astype(np.int32)
        bboxes = []
        for x, y in centers:
            h = h_hm[x, y]
            w = w_hm[x, y]
            x1, x2 = x - (h // 2), x + (h // 2)
            y1, y2 = y - (w // 2), y + (w // 2)
            bboxes.append([y1, x1, y2, x2])
        return np.clip(np.asarray(bboxes, dtype=np.int32), 0, img_size - 1)

    def nms(self, c_hm):
        batch_size, channels, height, width = c_hm.shape
        kernel_size = self.hparams.Train.maxpool_kernel_size
        scores, indices = F.max_pool2d(c_hm, kernel_size, return_indices=True)
        c_hm = F.max_unpool2d(scores, indices, kernel_size,
                              output_size=(batch_size, channels, height, width))
        return c_hm

    def get_pred_bboxes(self, c_hm, hw_hm):
        img_size = self.hparams.Train.img_size
        h_hm = (hw_hm[0] * img_size).cpu().numpy()
        w_hm = (hw_hm[1] * img_size).cpu().numpy()
        centers = torch.nonzero(c_hm[0] > 0.5).cpu().numpy().astype(np.int32)
        c_hm = c_hm[0].cpu().numpy()
        bboxes = []
        scores = []
        for x, y in centers:
            h = h_hm[x, y]
            w = w_hm[x, y]
            x1, x2 = y - (w // 2), y + (w // 2)
            y1, y2 = x - (h // 2), x + (h // 2)
            bboxes.append([x1, y1, x2, y2])
            scores.append(c_hm[x, y])
        sort_idx = np.argsort(scores)[::-1]
        bboxes = np.array(bboxes, dtype=np.int32)
        bboxes = bboxes[sort_idx]
        return np.clip(bboxes, 0, img_size - 1)
