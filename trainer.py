import json
import os
from pathlib import Path

import fsspec
import hydra
import lightning as L
import omegaconf
import torch
from tqdm import tqdm

import dataloader
import utils
import src.metrics as metrics
from lit_diffusion_base import resolve_diffusion_model
from lit_text_model import LitTextDDM
from utils import _register_resolvers


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _generate_samples_eval(model: LitTextDDM, config, tokenizer):
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()

    all_samples = []
    mauve_samples = [] if config.eval.compute_mauve else None
    for _ in tqdm(range(config.sampling.num_sample_batches)):
        samples = model.restore_model_and_sample(
            nfes=config.sampling.nfes, temperature=config.sampling.temperature
        )
        model.metrics.record_entropy(samples)
        text_samples = model.tokenizer.batch_decode(samples)
        if mauve_samples is not None:
            mauve_samples.extend(
                model.tokenizer.batch_decode(samples, skip_special_tokens=True)
            )
        if config.eval.compute_generative_perplexity:
            model.metrics.record_generative_perplexity(
                text_samples,
                config.model.length,
                device=model.device,
            )
        all_samples.extend(list(text_samples))

    generative_ppl = 0.0
    entropy = 0.0
    if config.eval.compute_generative_perplexity:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print("Generative perplexity:", generative_ppl)
        print("Sample entropy:", entropy)

    mauve_score = None
    if config.eval.compute_mauve:
        num_samples = (
            len(all_samples)
            if config.eval.mauve_num_samples is None
            else config.eval.mauve_num_samples
        )
        generated_texts = (
            mauve_samples[:num_samples] if mauve_samples is not None else []
        )

        _, valid_loader = dataloader.get_dataloaders(
            config,
            tokenizer,
            skip_train=True,
            valid_seed=config.seed,
        )
        reference_texts = []
        for batch in valid_loader:
            decoded = tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )
            reference_texts.extend(decoded)
            if len(reference_texts) >= num_samples:
                break
        reference_texts = reference_texts[:num_samples]

        mauve_output = metrics.compute_mauve_score(
            generated_texts=generated_texts,
            reference_texts=reference_texts,
            featurize_model_name_or_path=config.eval.mauve_model_name_or_path,
            device_id=config.eval.mauve_device_id,
            max_text_length=config.eval.mauve_max_text_length,
            num_buckets=config.eval.mauve_num_buckets,
            verbose=config.eval.mauve_verbose,
        )
        if hasattr(mauve_output, "mauve"):
            mauve_score = float(mauve_output.mauve)
        else:
            mauve_score = float(mauve_output["mauve"])
        print("MAUVE:", mauve_score)

    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, "w") as fp:
        json.dump(
            {
                "generative_ppl": generative_ppl,
                "entropy": entropy,
                "mauve": mauve_score,
                "algo_config": omegaconf.OmegaConf.to_container(config.algo),
                "sampling_config": omegaconf.OmegaConf.to_container(config.sampling),
                "generated_seqs": all_samples,
            },
            fp,
            indent=4,
        )
    print("Samples saved at:", samples_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)

    tokenizer = dataloader.get_tokenizer(config)
    diffusion_model = resolve_diffusion_model(config.algo.name)

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

    if config.mode == "train":
        train_loader, valid_loader = dataloader.get_dataloaders(config, tokenizer)
        model = LitTextDDM(
            config=config,
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            model_prediction=config.get("algo").get("model_prediction", None),
        )

        if config.training.finetune_path != "":
            logger.info(f"Loading backbone from {config.training.finetune_path}")
            model.load_backbone_from_checkpoint(config.training.finetune_path)

        resume_enabled = bool(config.checkpointing.resume_from_ckpt)
        resume_ckpt_path = config.checkpointing.resume_ckpt_path

        if resume_enabled and resume_ckpt_path is not None:
            resume_ckpt_path = str(resume_ckpt_path)
            if utils.fsspec_exists(resume_ckpt_path):
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

    if config.eval.checkpoint_path == "":
        raise ValueError("Set eval.checkpoint_path for non-train modes.")

    model = LitTextDDM(
        config=config,
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        model_prediction=config.get("algo").get("model_prediction", None),
    )
    model.load_backbone_from_checkpoint(config.eval.checkpoint_path)
    if config.eval.disable_ema:
        model.ema = None

    if config.mode == "ppl_eval":
        _, valid_loader = dataloader.get_dataloaders(
            config,
            tokenizer,
            skip_train=True,
            valid_seed=config.seed,
        )
        metrics_output = trainer.validate(model, valid_loader)
        metrics_output_path = str(config.eval.get("metrics_output_path", ""))
        if metrics_output_path != "":
            metrics_path = Path(metrics_output_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = _json_ready(metrics_output[0] if metrics_output else {})
            with metrics_path.open("w") as fp:
                json.dump(payload, fp, indent=2)
            print("Metrics saved at:", metrics_path)
    elif config.mode == "sample_eval":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        _generate_samples_eval(model, config, tokenizer)
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")


_register_resolvers()

if __name__ == "__main__":
    main()
