import argparse
import os.path as osp
import traceback  # Added for detailed error reporting
from typing import Dict

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from mmengine.config import Config, DictAction
from torch import Tensor

from Net.dataset.track_dataset import TrackDataModule
from Net.utils import (MODELS, generate_save_dir,
                           training_info)

def export_to_onnx(model, data_module, save_path):
    """Helper function to export the model to ONNX bypassing the dynamic_axes conversion bug."""
    print("\n>>> Starting ONNX Export...")
    model.eval()

    try:
        # Obtain and prepare sample input
        val_loader = data_module.val_dataloader()
        batch = next(iter(val_loader))
        input_sample = batch[0] if isinstance(batch, (list, tuple)) else batch

        if isinstance(input_sample, torch.Tensor):
            input_sample = input_sample.to(model.device)

        onnx_file = osp.join(save_path, 'model.onnx')

        # 1. We remove 'dynamic_axes' entirely for the first test.
        # This determines if the core model graph is exportable in Python 3.14.
        # 2. We pass input_sample directly without tuple-wrapping to avoid treespec errors.
        torch.onnx.export(
            model,
            input_sample,
            onnx_file,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        print(f">>> ONNX model successfully saved to: {onnx_file}")
        print(">>> NOTE: This export uses static shapes. Verify this works before re-enabling dynamic axes.\n")

    except Exception as e:
        import traceback
        print("\n" + "!"*30)
        print(">>> ONNX EXPORT FAILED")
        traceback.print_exc()
        print("!"*30 + "\n")
        print("Common Troubleshooting Tips:")
        print("1. Ensure your forward() method does not contain data-dependent control flow (e.g., if x.mean() > 0).")
        print("2. Check if all operations used in the model are supported by ONNX opset 12.")
        print("3. Verify that all tensors are on the same device as the model during export.")

def main(args: argparse.ArgumentParser, cfg: Config) -> None:
    training_info()
    torch.manual_seed(3407)
    save_dir: Dict = generate_save_dir(root='./runs',
                                       project=cfg.logger.project,
                                       name=cfg.logger.name)
    cfg.logger.name = save_dir['new_name']
    cfg.dump(osp.join(save_dir['config_dir'], 'config.py'))

    data_module = TrackDataModule(
        cfg, use_transform=cfg.data.transforms.use_transform)
    data_module.setup()

    model = MODELS.build(
        dict(type=cfg.trainer.type, cfg=cfg, save_dir=save_dir))

    # trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_monitor = ModelCheckpoint(
        dirpath=save_dir['weight_dir'],
        filename='{epoch}-{val_loss:.2f}-{val_MSE_dB:.2f}',
        mode='min',
        save_top_k=10,
        monitor='val_MSE_dB')
    callbacks = [lr_monitor, model_monitor]

    wandb_logger = WandbLogger(project=cfg.logger.project,
                               name=cfg.logger.name,
                               offline=cfg.logger.offline)

    trainer = Trainer(
        accelerator='cpu',
        max_epochs=cfg.trainer.epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        detect_anomaly=cfg.trainer.detect_anomaly,
        callbacks=callbacks,
        devices=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch if cfg.trainer.check_val_every_n_epoch is not None else 1
    )

    # Run training
    trainer.fit(model, datamodule=data_module)

    # Export to ONNX after training
    if cfg.get('export_onnx', True):
        export_to_onnx(model, data_module, save_dir['weight_dir'])


def parse_args():
    parser = argparse.ArgumentParser(
        prog='KalmanNet',
        description='Dataset, training and network parameters')
    parser.add_argument('--config',
                        '--cfg',
                        type=str,
                        metavar='config',
                        help='model and seq ')

    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config.')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(cfg)
    main(args, cfg)
