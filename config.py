from pathlib import Path
from omegaconf import OmegaConf

_root_dir = Path(__file__).parent

__model_type__ = 'faster_rcnn'
__config__ = dict(
    seed=42,
    model_type=__model_type__,
    train_csv=str(_root_dir / Path('data/trimmed_train.csv')),
    valid_csv=str(_root_dir / Path('data/validation.csv')),
    train_images_dir=str(_root_dir / Path('data/train')),
    Train=dict(
        checkpoint_dir=str(Path(f'checkpoints/{__model_type__}')),
        batch_size=2,
        lr=0.001,
        epochs=30,
        img_size=1024,
        steps_per_epoch=int(720 * (4 / 2)),
        scheduler='OneCycleLR',
        Schedulers=dict(
            interval='step',
            OneCycleLR=dict(
                pct_start=0.05,
                anneal_strategy='cos',
                div_factor=1e4
            ),
            NCycleLR=dict(
                n=3,
                steps_per_cycle=int(720 * (4 / 2) * 10),
                lr_factor=0.75,
                pct_start=0.05,
                anneal_strategy='cos',
                div_factor=1e4
            ),
            CyclicLR=dict(
                mode='triangular2',
                gamma=0.9
            ),
            ReduceLROnPlateau=dict(
                factor=0.1,
                patience=1
            )
        )
    )
)

Config = OmegaConf.create(__config__)
