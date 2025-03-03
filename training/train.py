#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The MedIT Solutions Kurman i Wspolnicy Sp. z o. o. Team.
# https://meditsolutions.pl

import os
import sys
import random
import logging
from typing import Tuple
from datetime import datetime

import torch
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AddedToken,
    get_cosine_schedule_with_warmup,
)
from transformers.data import DataCollatorForLanguageModeling
from transformers.trainer_pt_utils import get_parameter_names
from logging import getLogger

from one.modeling_one import OneForCausalLM, OneRMSNorm
from one.configuration_one import OneConfig
from config import load_config

# Configure logging
logger = getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Random seed set to {seed}")


def get_parameters_count(model) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_numel = 0
    trainable_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
            if p.requires_grad:
                trainable_numel += p.numel()
    return total_numel, trainable_numel


def train(config_path: str) -> None:
    """Train the model using the specified configuration.

    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)

    # Set random seed
    set_all_seeds(config.system.seed)

    # Log training parameters
    logger.info(
        f"Training {config.model.name} model with {config.model.tokenizer_name} tokenizer"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name, use_fast=True
    )

    # Ensure pad token is available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = AddedToken("<pad>", lstrip=True, rstrip=False)
        tokenizer.pad_token_id = tokenizer.add_tokens(tokenizer.pad_token)
        tokenizer.padding_side = "left"
        logger.info(
            f"Added pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}"
        )

    # Initialize or load model
    if config.model.name == "one":
        if config.model.checkpoint is not None:
            logger.info(f"Loading model from checkpoint: {config.model.checkpoint}")
            model = OneForCausalLM.from_pretrained(config.model.checkpoint)
        else:
            logger.info("Initializing a new model with the specified architecture")

            # Get dtype for model
            torch_dtype = config.get_torch_dtype()

            # Create model configuration
            model_configuration = OneConfig(
                vocab_size=len(tokenizer),
                num_hidden_layers=config.architecture.num_layers,
                hidden_size=config.architecture.hidden_size,
                transformer_dim=config.architecture.transformer_dim,
                intermediate_size=config.architecture.intermediate_size,
                num_attention_heads=config.architecture.num_heads,
                use_flash_attention=config.architecture.use_flash_attention,
                use_single_token=config.architecture.use_single_token,
                pad_token_id=tokenizer.pad_token_id,
                max_model_length=config.architecture.max_model_length,
                max_position_embeddings=config.architecture.max_model_length,
                torch_dtype=torch_dtype,
            )

            model = OneForCausalLM(model_configuration)
    else:
        raise ValueError(f"Model {config.model.name} not supported")

    # Count parameters
    total_params, trainable_params = get_parameters_count(model)
    logger.info(
        f"Model parameters: {total_params:,}, Trainable parameters: {trainable_params:,}"
    )

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset.train}")
    try:
        dataset_train = load_dataset(
            config.dataset.train,
            name=config.dataset.config_name,
            split="train",
            cache_dir=config.dataset.cache_dir,
        )

        # Process dataset
        datasets = [dataset_train.select_columns(["text"])]
        dataset = concatenate_datasets(datasets).shuffle(seed=config.system.seed)

        # Split into train and test
        test_size = config.dataset.train_test_split
        dataset = dataset.train_test_split(test_size=test_size, seed=config.system.seed)
        logger.info(
            f"Dataset split: {len(dataset['train']):,} training samples, {len(dataset['test']):,} test samples"
        )

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Create collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define tokenization function
    def tokenize_function(examples):
        tokens = tokenizer(
            [text + tokenizer.eos_token for text in examples["text"]],
            max_length=config.dataset.max_length,
            truncation=True,
            verbose=False,
            pad_to_multiple_of=64,
        )
        collated = collator(tokens["input_ids"])
        return collated

    # Process datasets
    logger.info("Tokenizing dataset...")
    skip = config.dataset.skip
    take = config.dataset.take
    train_samples = (
        len(dataset["train"]) - skip
        if take is None
        else min(take, len(dataset["train"]) - skip)
    )
    test_samples = len(dataset["test"])
    logger.info(
        f"Using {train_samples:,} training samples, {test_samples:,} test samples"
    )

    if config.model.trainer_checkpoint is None:
        train_ds = (
            dataset["train"]
            .to_iterable_dataset()
            .skip(skip)
            .take(take if take is not None else len(dataset["train"]))
            .map(tokenize_function, batched=True, batch_size=config.training.batch_size)
        )
        test_ds = (
            dataset["test"]
            .to_iterable_dataset()
            .map(tokenize_function, batched=True, batch_size=config.training.batch_size)
        )
    else:
        train_ds = (
            dataset["train"]
            .take(take if take is not None else len(dataset["train"]))
            .map(
                tokenize_function,
                batch_size=config.training.batch_size,
                batched=True,
                num_proc=config.dataset.num_proc,
            )
        )
        test_ds = dataset["test"].map(
            tokenize_function,
            batch_size=config.training.batch_size,
            batched=True,
            num_proc=config.dataset.num_proc,
        )

    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.model.trainer_checkpoint is None:
        output_dir = os.path.join(
            os.getcwd(),
            config.directories.output_dir,
            config.model.name,
            timestamp,
        )
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "/".join(config.model.trainer_checkpoint.split("/")[:-1])

    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    config.save(os.path.join(output_dir, "config.yaml"))

    # Calculate training steps
    batch_size = config.training.batch_size
    accumulation_iter = config.training.accumulation_iter
    epochs = config.training.epochs

    max_steps = (train_samples // (batch_size * accumulation_iter)) * epochs
    logger.info(f"Training for {max_steps:,} steps")

    # Prepare optimizer groups
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm, OneRMSNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n in decay_parameters
            ],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    # Create optimizer and scheduler
    optimizer = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=config.training.betas,
        eps=config.training.eps,
        lr=config.training.lr,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=max_steps,
    )

    # Create logging directory
    logging_dir = os.path.join(
        os.getcwd(),
        config.directories.logging_dir,
        config.model.name,
        timestamp,
    )

    # Setup training arguments
    training_args = TrainingArguments(
        num_train_epochs=epochs,
        learning_rate=config.training.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        disable_tqdm=False,
        push_to_hub=False,
        logging_strategy="steps",
        logging_dir=logging_dir,
        logging_steps=1,
        logging_nan_inf_filter=True,
        gradient_accumulation_steps=accumulation_iter,
        output_dir=output_dir,
        max_steps=max_steps,
        evaluation_strategy="steps",
        eval_steps=config.training.save_steps,
        eval_accumulation_steps=accumulation_iter,
        seed=config.system.seed,
        bf16=config.architecture.torch_dtype == "bfloat16",
        bf16_full_eval=config.architecture.torch_dtype == "bfloat16",
        neftune_noise_alpha=config.training.neftune_noise_alpha,
        save_total_limit=config.training.save_total_limit,
        include_num_input_tokens_seen=True,
        use_cpu=config.system.device == "cpu",
        save_safetensors=True,
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.with_format("torch"),
        eval_dataset=test_ds.with_format("torch"),
        optimizers=(optimizer, scheduler),
    )

    # Run training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=config.model.trainer_checkpoint)

    logger.info("Training complete")

    # Save final model
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MedIT ONE model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    args = parser.parse_args()
    train(args.config)
