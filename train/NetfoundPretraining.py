import math
import random
import os
import torch
import deepspeed
import utils
from dataclasses import field, dataclass
from typing import Optional
import gc 
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
)
import json
from WindowModel import NetFoundLanguageModelling
# from SlidingWindowModel import ChunkBasedTransformer
from NetFoundTrainer import NetfoundTrainer
from NetFoundDataCollator import DataCollatorWithMeta
from NetfoundConfig import NetfoundConfig
from optimized_tokenizer import NetFoundTokenizer
from utils import ModelArguments, CommonDataTrainingArguments, freeze, verify_checkpoint, \
    load_train_test_datasets, get_90_percent_cpu_count, initialize_model_with_deepspeed, get_logger, init_tbwriter,load_tokenized_dataset
from datasets import Dataset, concatenate_datasets
random.seed(42)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import pdb 

@dataclass
class PretrainingDataTrainingArguments(CommonDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    no_mlm: bool = field(
        default=False,
        metadata={"help": "no MLM loss function"},
    )
    no_swapped_bursts: bool = field(
        default=True,
        metadata={"help": "no swapped bursts loss function"},
    )
    swap_rate: Optional[float] = field(
        default=0.5,
        metadata={"help": "probability of swapping the burst in the flow during training"},
    )
    subflow_len: Optional[int] = field(
        default=-1,
        metadata={"help": "subflow length, -1 for no subflow"},
    )
    mlm_probability: float = field(
        default=0.30,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )


# def preprocess_logits_for_metrics(logits, _):
#     if isinstance(logits, tuple):
#         return tuple(i.argmax(dim=-1) for i in logits)
#     return logits.argmax(dim=-1)

def preprocess_logits_for_metrics(logits, _):
    # pdb.set_trace()
    # Target size for padding
    # target_size = 30000
    target_size = 9500
    # Handle tuples of logits
    if isinstance(logits, tuple):
        # Pad each tensor in the tuple to the target size along the third dimension
        padded_logits = [
            torch.nn.functional.pad(
                logit,
                (0, 0, 0, target_size - logit.size(2)),
                value=-100
            ) if logit is not None else None
            for logit in logits
        ]
        # Apply argmax to the padded logits
        return tuple(pad.argmax(dim=-1) if pad is not None else None for pad in padded_logits)

    # Handle single logits tensor
    if logits is not None:
        # Pad the logits tensor to the target size along the third dimension
        padded_logits = torch.nn.functional.pad(
            logits,
            (0, 0, 0, target_size - logits.size(2)),
            value=-100
        )
        return padded_logits.argmax(dim=-1)
    # pdb.set_trace()
    return None


def compute_metrics(eval_preds):
    all_preds, all_labels = eval_preds

    labels = all_labels[0] if isinstance(all_labels, tuple) else all_labels
    preds = all_preds[0] if isinstance(all_preds, tuple) else all_preds
    swappedBurstGTs = all_labels[1] if isinstance(all_labels, tuple) else None
    swappedBurstPreds = all_preds[1] if isinstance(all_preds, tuple) else None
    
    # pdb.set_trace()
    max_len = preds.shape[2]
    if labels.shape[1] < max_len:
        padding_size = max_len - labels.shape[1]
        labels = np.pad(labels, ((0, 0), (0, padding_size)), constant_values=-100)
    
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    
    # pdb.set_trace()
    # row, col = np.where((labels != -100))
    # token_map = {1: {}, 6: {}, 17: {}}
    # for i in range(len(row)):
    #     curr_proto = 6
    #     # pos_in_burst = col[i] % 109 - 1

    #     token_id = int(col[i] % 18)
 
    #     if token_id not in token_map[curr_proto]:
    #         token_map[curr_proto][token_id] = []
    #     token_map[curr_proto][token_id].append(np.array([labels[row[i]][col[i]], preds[row[i]][col[i]]]))
    # for prot in token_map:
    #     for tok in token_map[prot]:
    #         lab_preds = np.vstack(token_map[prot][tok])
    #         token_map[prot][tok] = float(accuracy_score(lab_preds[:, 0], lab_preds[:, 1]))
    # with open('/global/homes/h/haarika/pscratch/network-data-representation/model_output2/token_preds.json', 'w') as fp:
    #      json.dump(token_map, fp)
    return_metrics = {
        "macro_mlm_f1": f1_score(labels, preds, average="macro"),
        "macro_mlm_prec": precision_score(labels, preds, average="macro"),
        "macro_mlm_recall": recall_score(labels, preds, average="macro"),
        "weighted_mlm_f1": f1_score(labels, preds, average="weighted"),
        "weighted_mlm_prec": precision_score(labels, preds, average="weighted"),
        "weighted_mlm_recall": recall_score(labels, preds, average="weighted"),
        "mlm_acc": accuracy_score(labels, preds),
    }
    # if swappedBurstGTs is not None and swappedBurstPreds is not None:
    #     return_metrics.update(
    #         {
    #             "swapped_macro_pred_f1": f1_score(swappedBurstGTs, swappedBurstPreds, average="macro"),
    #             "swapped_macro_pred_prec": precision_score(
    #                 swappedBurstGTs, swappedBurstPreds, average="macro"
    #             ),
    #             "swapped_macro_pred_recall": recall_score(
    #                 swappedBurstGTs, swappedBurstPreds, average="macro"
    #             ),
    #             "swapped_weighted_pred_f1": f1_score(
    #                 swappedBurstGTs, swappedBurstPreds, average="weighted"
    #             ),
    #             "swapped_weighted_pred_prec": precision_score(
    #                 swappedBurstGTs, swappedBurstPreds, average="weighted"
    #             ),
    #             "swapped_weighted_pred_recall": recall_score(
    #                 swappedBurstGTs, swappedBurstPreds, average="weighted"
    #             ),
    #             "swapped_pred_acc": accuracy_score(swappedBurstGTs, swappedBurstPreds),
    #         }
    #     )

    
    
    
    return return_metrics


@record
def main():
    parser = HfArgumentParser(
        (ModelArguments, PretrainingDataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    utils.LOGGING_LEVEL = training_args.get_process_log_level()
    logger = get_logger(name=__name__)

    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    # pdb.set_trace()
    train_dataset, test_dataset = load_tokenized_dataset(logger, data_args)
    comb_dataset=concatenate_datasets([train_dataset,test_dataset])
    # pdb.set_trace()

    logger.warning("Tokenizing datasets")
    config = NetfoundConfig(
        num_hidden_layers=model_args.hidden_layers,
        no_meta=data_args.no_meta,
        flat=data_args.flat,
    )

    config.roformer = False
    config.limit_bursts = data_args.limit_bursts
    config.no_mlm = data_args.no_mlm
    if config.no_mlm:
        data_args.mlm_probability = 0.00001  # epsilon
    swap_rate = data_args.swap_rate
    config.no_swapped_bursts = data_args.no_swapped_bursts
    if config.no_swapped_bursts:
        swap_rate = 0
    config.name_or_path = model_args.model_name_or_path
    tokenizer = NetFoundTokenizer(config=config)

    data_collator = DataCollatorWithMeta(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        swap_rate=swap_rate
    )
    # pdb.set_trace()
    #train_dataset=train_dataset[:2]
    # train_dataset=train_dataset.select(range(32))
    
    # training_args = TrainingArguments(
    # output_dir="./results",
    # local_rank=-1,  # This disables distributed training (use -1 for single process)
    # num_train_epochs=3,
    # per_device_train_batch_size=8,
    # per_device_eval_batch_size=8,
    # logging_dir='./logs',
    # logging_steps=10,
    # )
    '''
    comb_dataset = comb_dataset.remove_columns(['labels'])
    comb_dataset = comb_dataset.remove_columns(['flow_duration'])
    '''
    # # train_dataset=train_dataset.select(range(50))
    # # train_dataset_new = train_dataset.map(tokenizer, batched=True,load_from_cache_file=True, num_proc=112,cache_file_name=None)
    '''
    batch_size = 10000
    
    # Get the total number of examples in the dataset
    total_examples = len(comb_dataset)
    # Loop through the dataset in increments of batch_size
    output_dir = "/global/homes/h/haarika/pscratch/network-data-representation/chunked_datasets/saved_chunked_dataset_01_08"
    pdb.set_trace()
    for start_idx in range(0, total_examples, batch_size):
        # Select a batch of examples
        end_idx = min(start_idx + batch_size, total_examples)
        
        # Process the batch
       
        # pdb.set_trace()
        # batch_name = f"processed_batch_{start_idx}_{end_idx}"
        # processed_batch.save_to_disk(f"/global/homes/h/haarika/pscratch/network-data-representation/saved_chunked_dataset/{batch_name}")
        batch_name = f"processed_batch_{start_idx}_{end_idx}"
        batch_path = os.path.join(output_dir, batch_name)

        # Check if the folder already exists; if not, save the batch
        if not os.path.exists(batch_path):
            batch = comb_dataset.select(range(start_idx, end_idx))
            batch=batch.select(range(5))
            # processed_batch = batch.map(tokenizer, batched=True, load_from_cache_file=False, num_proc=1, cache_file_name=None)
            processed_batch = batch.map(tokenizer, batched=True, load_from_cache_file=False, num_proc=1, cache_file_name=None)
            processed_batch = processed_batch.remove_columns(['directions'])
            processed_batch = processed_batch.rename_column("directions_tok","directions")
            # pdb.set_trace()
            processed_batch.save_to_disk(batch_path)
            del processed_batch
            del batch
        else:
            print(f"Skipping save for {batch_name}, folder already exists.")
    
        gc.collect()
    '''
    # train_dataset=train_dataset.select(range(6))
    # train_dataset_new = train_dataset.map(tokenizer, batched=True,load_from_cache_file=False, num_proc=112,cache_file_name=None)
    # pdb.set_trace()

    # test_dataset_new = test_dataset.map(tokenizer, batched=True, num_proc= 112)
    # train_dataset = train_dataset.select([0]) 
 
    # if model_args.model_name_or_path is not None and os.path.exists(
    #         model_args.model_name_or_path
    # ):
    #     logger.warning(f"Using weights from {model_args.model_name_or_path}")
    #     get_model = lambda: freeze(NetFoundLanguageModelling.from_pretrained(
    #         model_args.model_name_or_path, config=config
    #     ), model_args)
    # else:
    # pdb.set_trace()
    get_model = lambda: NetFoundLanguageModelling(config=config)


    if training_args.deepspeed:
        print("Initilizing Deepspeed NEW !")
        model, optimizers = initialize_model_with_deepspeed(logger, training_args, get_model)
        # model, optimizer, _, _ = deepspeed.initialize(
        # model= NetFoundLanguageModelling(config=config),
        # optimizer=None,
        # config=training_args.deepspeed,
        # model_parameters=model.parameters()
        # )
    else:
        logger.warning("Initializing model")
        model = get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.99
        )
        optimizers = (optimizer, lr_scheduler)

    # verify_checkpoint(logger, training_args)

    trainer = NetfoundTrainer(
        label_names=["swappedLabels"],
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator
    )
    init_tbwriter(trainer)

    checkpoint = None
    # if training_args.do_train:
    # training_args = TrainingArguments(
    # output_dir="./results",
    # local_rank=-1,  # Ensure single-process
    # num_train_epochs=3,
    # per_device_train_batch_size=8,
    # logging_dir='./logs'
    # )

    if training_args.do_train:
        logger.warning("*** Train ***")
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # pdb.set_trace()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.warning("*** Evaluate ***")
        metrics = trainer.evaluate()
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
