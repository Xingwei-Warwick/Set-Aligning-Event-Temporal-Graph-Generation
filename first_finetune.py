import json
from transformers import HfArgumentParser, AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from tqdm import tqdm, trange
from datasets import Dataset
import numpy as np
import random
from itertools import permutations
from argparse import ArgumentParser
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


MAX_TARGET_LENGTH = 1024


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    train_from_checkpoint: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    max_target_length: int = field(default=512)


def get_tokenizer_and_data(training_args, model_args, data_args, max_target_length=1024):
    def _preprocess_function(examples):
        inputs = [doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=training_args.model_max_length, truncation=True, padding=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True, padding=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    with open(data_args.data_path, 'r') as f:
        train_data = json.loads(f.read())
    
    print("Training documents:")
    print(len(train_data))
    input_dict = {
        'document': [train_data[d]['document'] for d in train_data],
        'summary': [train_data[d]['target'] for d in train_data],
        'id': list(range(len(train_data.keys())))
    }

    train_dataset = Dataset.from_dict(input_dict)

    tokenized_datasets = train_dataset.map(_preprocess_function, batched=True)

    return tokenizer, tokenized_datasets


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    save_name = training_args.output_dir

    tokenizer, tokenized_datasets = get_tokenizer_and_data(training_args, model_args, data_args,training_args.max_target_length)

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if model_args.train_from_checkpoint:
        trainer.train(model_args.model_name_or_path)
    else:
        trainer.train()
        
    trainer.save_state()
    trainer.save_model(f"{save_name}-final")

    print(training_args)
    print(model_args)
    print(data_args)


if __name__ == "__main__":
    train()

