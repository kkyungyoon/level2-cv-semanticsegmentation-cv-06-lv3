import argparse
import pytorch_lightning as pl

from src.logger.custom_logger import _get_logger
from src.logger.base_logger import log_configs_to_file
from src.callbacks.base_callback import _get_callbacks
from src.utils.data_utils import load_yaml_config
from src.data.custom_datamodules.xray_datamodule import XRayDataModule
from src.plmodules.smp_module import SmpModule

def main(config_path):

    # load config file
    print(f"... Loading config from {config_path} ...")
    config = load_yaml_config(config_path=config_path)
    util_config = load_yaml_config(config["path"].get("util", ""))
    train_config = load_yaml_config(config["path"].get("train", ""))

    print(f"... {config_path} file loaded ...")

    # check whether auto mixed precision is used
    if "amp" in util_config and util_config["amp"].get("enabled", False):
        print(f'... Use auto mixed precision (amp) ...')
        amp = "bf16"
    else:
        amp = None

    # Load datamodule
    data_module = XRayDataModule(config=config)

    # Setup datamodule
    data_module.setup()

    # Load model
    model = SmpModule(config=config)
    
    # Setup logger
    logger = _get_logger(config=train_config.get("logger", {}))

    # log your setting in txt
    log_configs_to_file(
        config=config, 
        output_dir=[
            train_config["logger"].get("project", ""),
            train_config["logger"].get("name", "")
            ]
        )

    # Setup callback
    callbacks = _get_callbacks(
        train_config.get("callbacks", {}),
        train_config.get("logger", {}),
    )

    # Load trainer with your settings
    # callbacks: earlystop, val_interval
    # loggers: tensorboard, wandb
    # precision: use amp or not
    # accumulate_grad_batches: default (16 same as the number of groups in gn)
    trainer = pl.Trainer(
        **train_config["trainer"],
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
    args = parser.parse_args()
    main(args.config, args.amp)



