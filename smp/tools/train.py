import argparse
import pytorch_lightning as pl

from src.logger.custom_logger import _get_logger
from src.logger.base_logger import log_configs_to_tensorboard
from src.callbacks.base_callback import _get_callbacks
from src.utils.data_utils import load_yaml_config
from src.data.custom_datamodules.xray_datamodule import XRayDataModule
from src.plmodules.smp_module import SmpModule

def main(config_path, amp=False):

    # load config file
    print(f"Loading config from {config_path} ...")
    config = load_yaml_config(config_path=config_path)
    print(f"{config_path} file loaded.")

    # check whether auto mix precision is used
    amp = None

    if amp:
        print(f'Use auto mix precision (amp).')
        amp = "bf16"

    aug_path = config["path"]["augmentation"]
    data_path = config["path"]["data"]
    model_path = config["path"]["model"]

    # Load datamodule
    data_module = XRayDataModule(
        data_config_path=data_path,
        augmentation_config_path=aug_path
    )

    # Setup datamodule
    data_module.setup()

    # Load model
    model = SmpModule(
        train_config_path=config_path,
        model_config_path=model_path
    )
    
    # Setup logger
    logger = _get_logger(
        config["logger"]
    )

    # log your setting in tb
    log_configs_to_tensorboard(logger[0], config)

    # Setup callback
    callbacks = _get_callbacks(
        config["callbacks"],
        config["logger"]
    )

    # Load trainer with your settings
    # callbacks: earlystop, val_interval
    # loggers: tensorboard, wandb
    # precision: use amp or not
    # accumulate_grad_batches: default (16 same as the number of groups in gn)
    trainer = pl.Trainer(
        **config["trainer"],
        callbacks=callbacks,
        logger=logger,
        precision=amp,
        accumulate_grad_batches=16,
    )

    # Fit
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--amp", action='store_true', help="Whether to use amp"
    )
    args = parser.parse_args()
    main(args.config, args.amp)



