import pytorch_lightning as pl
import torch
from stable_audio_tools.models import create_model_from_config
import json


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint
    ) -> None:
        checkpoint["model_config"] = self.model_config


class SigtermCallback(pl.Callback):
    def __init__(self, save_on_sigterm=True):
        self.save_on_sigterm = save_on_sigterm

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        import signal

        def handle_sigterm(signum, frame):
            if self.save_on_sigterm:
                monitor_candidates = self.ckpt_callback._monitor_candidates(trainer)
                self.ckpt_callback._save_last_checkpoint(trainer, monitor_candidates)
                self.ckpt_callback._save_topk_checkpoint(trainer, monitor_candidates)

        # This registers the hook
        signal.signal(signal.SIGTERM, handle_sigterm)


class ModifyLRCallback(pl.Callback):
    def __init__(self, lr):
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        optimizers = pl_module.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = self.lr
        print(f"Set lr to {self.lr}")


class ModifyPretransformCallback(pl.Callback):
    def __init__(self, pretransform_ckpt_path):
        self.pretransform_ckpt_path = pretransform_ckpt_path

    def on_train_start(self, trainer, pl_module):
        with open(
            "/home/diont/htools_training/used_configs_Wsl/autoencoder/dac_512_32_vae_stereo_44k.json"
        ) as f:
            model_config = json.load(f)
        pretransform = create_model_from_config(model_config)
        pretransform.load_state_dict(
            torch.load(self.pretransform_ckpt_path)["state_dict"]
        )
        pl_module.pretransform = pretransform

        del pretransform

        import gc

        torch.cuda.empty_cache()

        gc.collect()

        print("Loaded pretransform from checkpoint")


class ExceptionCallback(pl.Callback):
    def __init__(self, ckpt_callback, save_on_exception=False):
        self.save_on_exception = save_on_exception
        self.ckpt_callback = ckpt_callback

    def on_exception(self, trainer, module, err):
        if self.save_on_exception:
            monitor_candidates = self.ckpt_callback._monitor_candidates(trainer)
            self.ckpt_callback._save_last_checkpoint(trainer, monitor_candidates)
            self.ckpt_callback._save_topk_checkpoint(trainer, monitor_candidates)
        print(f"{type(err).__name__}: {err}")
