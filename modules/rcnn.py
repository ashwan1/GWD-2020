import numpy as np
import torch
from torch.nn import Sequential
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR, ReduceLROnPlateau
import torchvision
from pytorch_lightning import LightningModule
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from custom.lr_schedulers.n_cycle import NCycleLR
from custom.optimizers import RangerLars
from utils.data import get_image_with_results
from utils.losses_n_metrics import calculate_image_precision


# noinspection PyAbstractClass
class FasterRCNNResnet50FPN(LightningModule):
    def __init__(self, conf=None, *args, **kwargs):
        super().__init__()
        self.hparams = conf
        resnest = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        backbone = Sequential(*list(resnest.children())[:-3])
        backbone.out_channels = 1024

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=[7, 7],
                                                        sampling_ratio=2)
        self.model = FasterRCNN(backbone,
                                num_classes=2,
                                rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        scheduler = None
        params = [p for p in self.model.parameters() if p.requires_grad]
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

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # FasterRCNN model returns dict with classification and regression loss
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        lr = self.trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']
        return {'loss': total_loss, 'log': loss_dict, 'progress_bar': {'lr': lr}}

    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)
        precisions = np.zeros((self.hparams.Train.batch_size,))
        for i, output in enumerate(outputs):
            scores = torch.argsort(output['scores'], descending=True)
            boxes = output['boxes'][scores].data.cpu().numpy().astype(np.int32)
            gt = targets[i]['boxes'].data.cpu().numpy().astype(np.int32)
            precision = calculate_image_precision(gt, boxes)
            precisions[i] = precision
        return {'val_loss': torch.tensor([1 - precisions], dtype=torch.float32, device='cuda'),
                'val_map': torch.tensor([precisions], dtype=torch.float32, device='cuda')}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_map'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_map': avg_precision}
        return {'avg_val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            images, targets, _ = batch
            targets = [{k: v for k, v in t.items()} for t in targets]
            outputs = self.model(images, targets)
            for i, output in enumerate(outputs):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                scores = torch.argsort(output['scores'], descending=True)
                boxes = output['boxes'][scores].data.cpu().numpy().astype(np.int32)
                gt = targets[i]['boxes'].data.cpu().numpy().astype(np.int32)
                results = get_image_with_results(image, boxes, gt)
                self.logger.experiment.add_image(f"bb_test(RED: Predicted; BLUE: Ground-truth)/image{i}",
                                                 results, dataformats='HWC')
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        test_results = self.validation_epoch_end(outputs)
        test_results = {k.replace('val', 'test'): v.cpu().numpy().tolist() for k, v in test_results['log'].items()}
        return {'log': test_results}
