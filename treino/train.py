import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import json
import pathlib
from pathlib import Path

model_checkpoint = "unicamp-dl/ptt5-base-portuguese-vocab"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

max_input_length = 384  # 512
max_target_length = 32  # 32
val_max_answer_length = max_target_length

pad_to_max_length = True
padding = "max_length" if pad_to_max_length else False
ignore_pad_token_for_loss = True

max_seq_length = min(max_input_length, tokenizer.model_max_length)
generation_max_length = None
max_eval_samples = None

version_2_with_negative = False  # squad 1.1

answer_column = "answers"

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# do training and evaluation
do_train = True
do_eval = True

# batch
batch_size = 4
gradient_accumulation_steps = 3
per_device_train_batch_size = batch_size
per_device_eval_batch_size = per_device_train_batch_size * 16

# LR, wd, epochs
learning_rate = 1e-4
weight_decay = 0.01
num_train_epochs = 10
fp16 = True

# logs
logging_strategy = "steps"
logging_first_step = True
logging_steps = 3000  # if logging_strategy = "steps"
eval_steps = logging_steps

# checkpoints
evaluation_strategy = logging_strategy
save_strategy = logging_strategy
save_steps = logging_steps
save_total_limit = 3

# best model
load_best_model_at_end = True
metric_for_best_model = "f1"  # "loss"
if metric_for_best_model == "loss":
    greater_is_better = False
else:
    greater_is_better = True

# evaluation
num_beams = 1

# folders
model_name = model_checkpoint.split("/")[-1]
folder_model = f'e{num_train_epochs}_lr{learning_rate}'
output_dir = f'/modelos/{str(model_name)}/checkpoints/{folder_model}'
Path(output_dir).mkdir(parents=True, exist_ok=True)  # python 3.5 above
logging_dir = f'/log/{str(model_name)}/logs/{folder_model}'
Path(logging_dir).mkdir(parents=True, exist_ok=True)  # python 3.5 above

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    do_train=do_train,
    do_eval=do_eval,
    evaluation_strategy=evaluation_strategy,
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    weight_decay=weight_decay,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    save_strategy=save_strategy,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    logging_dir=logging_dir,  # directory for storing logs
    logging_strategy=logging_strategy,
    logging_steps=logging_steps,  # if logging_strategy = "steps"
    # fp16=fp16,
    push_to_hub=False,
)

raw_datasets = load_dataset('json',
                        data_files={'train': 'datasets/pt_squad-train-v1.1.json', 'validation': 'datasets/pt_squad-dev-v1.1.json'},
                        field='data')


def preprocess_squad_batch(examples):
  targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in examples['answers']]
  return examples['input_ids'], targets

# train preprocessing
def preprocess_train_function(examples):

    inputs, targets = preprocess_squad_batch(examples)

    # inputs = [prefix + doc for doc in inputs]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
      labels["input_ids"] = [
                             [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                             ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# Validation preprocessing
def preprocess_validation_function(examples):
    inputs, targets = preprocess_squad_batch(examples)

    # inputs = [prefix + doc for doc in inputs]
    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True,
                             return_overflowing_tokens=True,
                             return_offsets_mapping=True, )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    model_inputs["example_id"] = []
    labels_mapping = {}
    labels_mapping['input_ids'] = []
    labels_mapping['attention_mask'] = []

    for i in range(len(model_inputs["input_ids"])):
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        model_inputs["example_id"].append(examples["id"][sample_index])
        labels_mapping['input_ids'].append(labels['input_ids'][sample_index])
        labels_mapping['attention_mask'].append(labels['attention_mask'][sample_index])

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_mapping["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = raw_datasets["train"]
eval_examples = raw_datasets["validation"]

column_names = raw_datasets["train"].column_names

# Create train feature from dataset
with training_args.main_process_first(desc="train dataset map pre-processing"):
  train_dataset = train_dataset.map(
      preprocess_train_function,
      batched=True,
      num_proc=None,
      remove_columns=column_names,
      load_from_cache_file=True,
      desc="Running tokenizer on train dataset",
      )


column_names = raw_datasets["validation"].column_names

with training_args.main_process_first(desc="validation dataset map pre-processing"):
  eval_dataset = eval_examples.map(
      preprocess_validation_function,
      batched=True,
      num_proc=None,
      remove_columns=column_names,
      load_from_cache_file=True,
      desc="Running tokenizer on validation dataset",
      )



# set format for pytorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'example_id', 'offset_mapping'])

# save
tokenized_datasets_dir = f'datasets/{str(model_name)}/tokenized_datasets/train/'
train_dataset.save_to_disk(tokenized_datasets_dir)
tokenized_datasets_dir = f'datasets/{str(model_name)}/tokenized_datasets/validation/'
eval_dataset.save_to_disk(tokenized_datasets_dir)

# load
tokenized_datasets_dir = f'datasets/{str(model_name)}/tokenized_datasets/train/'
train_dataset = load_from_disk(tokenized_datasets_dir)
tokenized_datasets_dir = f'datasets/{str(model_name)}/tokenized_datasets/validation/'
eval_dataset = load_from_disk(tokenized_datasets_dir)


# Data collator
label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
    )


metric = load_metric("squad_v2" if version_2_with_negative else "squad")

def compute_metrics(p):
  return metric.compute(predictions=p.predictions, references=p.label_ids)


# Post-processing:
def post_processing_function(examples, features, outputs, stage="eval"):
    # Decode the predicted tokens.
    preds = outputs.predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    # print('example_id_to_index:',example_id_to_index)
    # print('features:',features)
    feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
    predictions = {}
    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # This is the index of the feature associated to the current example.
        feature_index = feature_per_example[example_index]
        predictions[example["id"]] = decoded_preds[feature_index]

    # Format the result to the format the metric expects.
    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column]} for ex in examples]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


early_stopping_patience = save_total_limit

# Initialize our Trainer
trainer = QuestionAnsweringSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if do_train else None, #.shard(num_shards=400, index=0)
    eval_dataset=eval_dataset if do_eval else None, #.shard(num_shards=400, index=0)
    eval_examples=eval_examples if do_eval else None, #.shard(num_shards=400, index=0)
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    post_process_function=post_processing_function,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )


trainer.train()

dir_checkpoint = str(f'/modelos/{str(model_name)}/checkpoints/{folder_model}/checkpoint-12000')
trainer.train(dir_checkpoint)

dir_checkpoint = str(f'/modelos/{str(model_name)}/checkpoints/{folder_model}/checkpoint-21000')
trainer.train(dir_checkpoint)

dir_checkpoint = str(f'/modelos/{str(model_name)}/checkpoints/{folder_model}/checkpoint-27000')
trainer.train(dir_checkpoint)


max_length=32
num_beams=1
early_stopping=True

results = {}
max_length = (generation_max_length if generation_max_length is not None else val_max_answer_length)
num_beams = num_beams if num_beams is not None else generation_num_beams

if do_eval:
    print("*** Evaluate ***")
    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    max_eval_samples = max_eval_samples if max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)

    eval_dir = f'/modelos/{str(model_name)}/eval_metrics/{folder_model}'
    Path(eval_dir).mkdir(parents=True, exist_ok=True)  # python 3.5 above
    trainer.save_metrics(eval_dir, metrics)

model_name = model_checkpoint.split("/")[-1]
model_dir = f'/modelos/{str(model_name)}/models/{folder_model}'
trainer.save_model(model_dir)

