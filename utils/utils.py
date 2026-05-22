"""Shared utility helpers for training and data loading."""

import logging

import fsspec
import lightning
import torch
from timm.scheduler import CosineLRScheduler
import os

import omegaconf


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Create directories in a way compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


class LRHalveScheduler:
    def __init__(self, warmup_steps, n_halve_steps):
        self.warmup_steps = warmup_steps
        self.n_halve_steps = n_halve_steps

    def __call__(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        return 0.5 ** (
            (current_step - self.warmup_steps) // self.n_halve_steps
        )


class CosineDecayWarmupLRScheduler(
    CosineLRScheduler, torch.optim.lr_scheduler._LRScheduler
):
    """Wrap `timm.scheduler.CosineLRScheduler` with a PyTorch-style step API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initialize a multi-process-safe logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for method_name in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(
            logger,
            method_name,
            lightning.pytorch.utilities.rank_zero_only(
                getattr(logger, method_name)
            ),
        )

    return logger

    # Computing the integral over [-5, 1] can be slow,
    # so one might prefer splitting it into `num_partitions`
    # bins and compute each separately and merge them later.
    _cache_prob_usdm_in_partition(
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
        vocab_size=args.vocab_size,
        log10_num_points=args.log10_num_points)
    
    test_cache_prob_usdm_in_partition(
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
        vocab_size=args.vocab_size,
        log10_num_points=args.log10_num_points)


def sample_categorical(categorical_probs, generator=None):
    probs = categorical_probs.to(torch.float64)
    gumbel = -torch.log(
        -torch.log(
            torch.rand(
                probs.shape,
                dtype=probs.dtype,
                device=probs.device,
                generator=generator,
            )
        )
    )
    return (probs.log() + gumbel).argmax(dim=-1)


def _visible_device_count():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",")]
        devices = [device for device in devices if device and device != "-1"]
        if devices:
            return len(devices)
    return torch.cuda.device_count()


def _register_resolvers():
    resolvers = {
        "cwd": os.getcwd,
        "device_count": torch.cuda.device_count,
        "visible_device_count": _visible_device_count,
        "eval": eval,
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, fn in resolvers.items():
        try:
            omegaconf.OmegaConf.register_new_resolver(name, fn)
        except ValueError:
            pass
