import logging
import os

import fsspec
import hydra
import lightning as L
import omegaconf
from torch.utils.data import DataLoader

from lit_diffusion_base import resolve_diffusion_model
from lit_sudoku_model import LitSudokuDDM
from utils.sudoku_utils import CustomTokenizer, SudokuDataset, collate
from utils.utils import _register_resolvers


def fsspec_exists(filename):
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def make_loader(config, tokenizer, split, dataset_cls, collate_fn, path_getter, path_arg):
    if split == "train":
        data_path = path_getter(config, "train")
        max_samples = config.data.max_train_samples
        batch_size = config.loader.batch_size
        num_workers = config.loader.train_num_workers
        shuffle = True
    elif split == "valid":
        data_path = path_getter(config, "valid")
        max_samples = config.data.max_valid_samples
        batch_size = config.loader.eval_batch_size
        num_workers = config.loader.eval_num_workers
        shuffle = False
    else:
        raise ValueError(f"Unsupported split: {split}")

    dataset = dataset_cls(
        **{
            path_arg: data_path,
            "tokenizer": tokenizer,
            "cutoff_len": config.data.cutoff_len,
            "max_samples": max_samples,
        }
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.loader.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )


def absolutize_common_config(config):
    with omegaconf.open_dict(config):
        config.data.tokenizer_config_path = hydra.utils.to_absolute_path(
            config.data.tokenizer_config_path
        )


def run_task(
    config,
    *,
    model_cls,
    model_kwargs=None,
    tokenizer_cls,
    dataset_cls,
    collate_fn,
    path_getter,
    path_arg,
):
    L.seed_everything(config.seed)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    absolutize_common_config(config)
    tokenizer = tokenizer_cls.from_pretrained(config.data.tokenizer_config_path)

    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb,
        )

    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    if model_kwargs is None:
        model_kwargs = {}
    model = model_cls(config=config, tokenizer=tokenizer, **model_kwargs)

    if config.mode == "train":
        train_loader = make_loader(
            config,
            tokenizer,
            "train",
            dataset_cls,
            collate_fn,
            path_getter,
            path_arg,
        )
        valid_loader = make_loader(
            config,
            tokenizer,
            "valid",
            dataset_cls,
            collate_fn,
            path_getter,
            path_arg,
        )

        resume_enabled = bool(config.checkpointing.resume_from_ckpt)
        resume_ckpt_path = config.checkpointing.resume_ckpt_path
        if resume_enabled and resume_ckpt_path is not None:
            resume_ckpt_path = str(resume_ckpt_path)
            if fsspec_exists(resume_ckpt_path):
                ckpt_path = resume_ckpt_path
                logger.info(f"Resuming training from checkpoint: {ckpt_path}")
            else:
                ckpt_path = None
                logger.warning(
                    "Resume requested, but checkpoint not found: "
                    f"{resume_ckpt_path} (cwd={os.getcwd()}). Starting from scratch."
                )
        elif resume_enabled:
            ckpt_path = None
            logger.warning(
                "Resume requested, but checkpointing.resume_ckpt_path is null. "
                "Starting from scratch."
            )
        else:
            ckpt_path = None
            logger.info(
                "Resume disabled (checkpointing.resume_from_ckpt=false). "
                "Starting from scratch."
            )

        trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt_path)
        return

    if config.mode == "val":
        if config.eval.checkpoint_path == "":
            raise ValueError("Set eval.checkpoint_path for val mode.")
        valid_loader = make_loader(
            config,
            tokenizer,
            "valid",
            dataset_cls,
            collate_fn,
            path_getter,
            path_arg,
        )
        trainer.validate(model, valid_loader, ckpt_path=config.eval.checkpoint_path)
        return

    raise ValueError(f"Unsupported mode: {config.mode}")


def _path_getter(config, split):
    with omegaconf.open_dict(config):
        if split == "train":
            config.data.train_csv = hydra.utils.to_absolute_path(config.data.train_csv)
            return config.data.train_csv
        if split == "valid":
            config.data.valid_csv = hydra.utils.to_absolute_path(config.data.valid_csv)
            return config.data.valid_csv
    raise ValueError(f"Unsupported split: {split}")


@hydra.main(version_base=None, config_path="configs", config_name="sudoku_config")
def main(config):
    run_task(
        config,
        model_cls=LitSudokuDDM,
        model_kwargs={
            "diffusion_model": resolve_diffusion_model(config.algo.name),
            "model_prediction": config.get("algo").get("model_prediction", None),
        },
        tokenizer_cls=CustomTokenizer,
        dataset_cls=SudokuDataset,
        collate_fn=collate,
        path_getter=_path_getter,
        path_arg="csv_path",
    )


_register_resolvers()

if __name__ == "__main__":
    main()
