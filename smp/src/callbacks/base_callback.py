from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

def _get_callbacks(config, logger_config):
    callbacks = []

    callbacks.append(
        LearningRateMonitor(logging_interval='epoch')
    )

    if config.get("callbacks", {}).get("model_checkpoint", {}).get("enabled", False):
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"./logs/{logger_config['name']}/checkpoints/",
                filename="{epoch:02d}-{avg_dice_score:.3f}",
                **config["callbacks"]["model_checkpoint"]["params"]
            )
        )

    if config.get("callbacks", {}).get("earlystopping", {}).get("enabled", False):
        callbacks.append(
            EarlyStopping(
                **config["callbacks"]["earlystopping"]["params"]
            )
        )

    return callbacks