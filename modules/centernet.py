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
        self.layer4 = resnest.layer4  # 2048x32x32
        self.conv2 = ConvBnRelu(2048, 1024, 3, padding=1)
        self.conv3 = ConvBnRelu(1024, 512, 5, padding=2)
        self.conv4 = ConvBnRelu(512, 256, 5, padding=2)
        self.conv5 = ConvBnRelu(256, 64, 3, padding=1)
        self.conv6 = ConvBnRelu(64, 32, 3, padding=1)
        self.conv7 = torch.nn.Conv2d(32, 1, 1)
        self.conv8 = torch.nn.Conv2d(32, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        xr = self.relu(x)
        x = self.maxpool(xr)
        xl1 = self.layer1(x)
        xl2 = self.layer2(xl1)
        xl3 = self.layer3(xl2)
        xl4 = self.layer4(xl3)
        x = self.conv2(xl4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + xl3
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + xl2
        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + xl1
        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + xr
        x = self.conv6(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x_center = self.conv7(x)
        x_center = F.sigmoid(x_center)
        x_hw = self.conv8(x)
        return x_center, x_hw

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
    def get_losses(self, center_hm, hw_hm, targets):
        hw_hm = torch.clamp(hw_hm, 0, 1)
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
        center_hm, hw_hm = self.forward(images)
        center_hm_loss, hw_hm_loss = self.get_losses(center_hm, hw_hm, targets)
        loss = center_hm_loss + hw_hm_loss
        lr = self.trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']
        loss_dict = {
            'chm_loss': center_hm_loss,
            'hwhm_loss': hw_hm_loss
        }
        return {'loss': loss, 'log': loss_dict, 'progress_bar': {'lr': lr}}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        center_hm, hw_hm = self.forward(images)
        center_hm_loss, hw_hm_loss = self.get_losses(center_hm, hw_hm, targets)
        loss = center_hm_loss + hw_hm_loss
        precisions = np.zeros((self.hparams.Train.batch_size,))
        center_hm = self.nms(center_hm)
        for i in range(targets.shape[0]):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            gt_bboxes = self.hm2bboxes(targets[i], 0.9999)
            predictions = torch.cat([center_hm[i], hw_hm[i]], 0)
            pred_bboxes = self.hm2bboxes(predictions, 0.6)
            if batch_idx == 0:
                write_img_to_disk(img, gt_bboxes, file_path=Path.cwd() / f'gt_{batch_idx}_{i}.jpg')
                write_img_to_disk(img, pred_bboxes, file_path=Path.cwd() / f'pred_{batch_idx}_{i}.jpg')
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
    def hm2bboxes(self, target, threshold):
        img_size = target[0].shape[0]
        hw_hm = target[1:].cpu().numpy()
        centers = torch.nonzero(target[0] > threshold).cpu().numpy().astype(np.int32)
        c_hm = target[0].cpu().numpy()
        bboxes = []
        scores = []
        for row, col in centers:
            h, w = hw_hm[:, row, col]
            y_min, x_min = row - (h // 2), col - (w // 2)
            y_max, x_max = y_min + h, x_min + w
            bboxes.append([x_min, y_min, x_max, y_max])
            scores.append(c_hm[row, col])
        sort_idx = np.argsort(scores)[::-1]
        bboxes = np.array(bboxes, dtype=np.int32)
        bboxes = bboxes[sort_idx]
        return np.clip(bboxes, 0, img_size - 1)

    def nms(self, c_hm):
        kernel = self.hparams.Train.maxpool_kernel_size
        pad = (kernel - 1) // 2
        c_hm_max = F.max_pool2d(c_hm, (kernel, kernel), stride=1, padding=pad)
        keep = (c_hm_max == c_hm).float()
        return c_hm * keep
