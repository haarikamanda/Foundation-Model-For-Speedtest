import os
import torch
import numpy as np
import utils
from dataclasses import field, dataclass
from typing import Optional
from copy import deepcopy

from transformers import (
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    classification_report, confusion_matrix
)

from NetFoundDataCollator import DataCollatorForFlowClassification
from NetFoundModels import NetfoundFinetuningModel, NetfoundNoPTM
from NetFoundTrainer import NetfoundTrainer
from NetfoundConfig import NetfoundConfig
from NetfoundTokenizer import NetFoundTokenizer
from utils import ModelArguments, CommonDataTrainingArguments, freeze, get_logger, verify_checkpoint, \
    load_train_test_datasets, get_90_percent_cpu_count, initialize_model_with_deepspeed, init_tbwriter


logger = get_logger(name=__name__)


@dataclass
class FineTuningDataTrainingArguments(CommonDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    num_labels: int = field(metadata={"help": "number of classes in the datasets"}, default=None)
    problem_type: Optional[str] = field(
        default=None,
        metadata={"help": "Override regression or classification task"},
    )
    p_val: float = field(
        default=0,
        metadata={
            "help": "noise rate"
        },
    )


def regression_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = p.label_ids.astype(int)
    return {"loss": np.mean(np.absolute((logits - label_ids)))}


def classif_metrics(p: EvalPrediction, num_classes):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = p.label_ids.astype(int)
    weighted_f1 = f1_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    weighted_prec = precision_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true=label_ids, y_pred=logits.argmax(axis=1))
    logger.warning(classification_report(label_ids, logits.argmax(axis=1), digits=5))
    logger.warning(confusion_matrix(label_ids, logits.argmax(axis=1)))
    if num_classes > 3:
        logger.warning(f"top2:{top_k_accuracy_score(label_ids, logits, k=2, labels=np.arange(num_classes))}")
        logger.warning(f"top3:{top_k_accuracy_score(label_ids, logits, k=3, labels=np.arange(num_classes))}")
    return {
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "weighted_prec: ": weighted_prec,
        "weighted_recall": weighted_recall,
    }


def main():
    parser = HfArgumentParser(
        (ModelArguments, FineTuningDataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    log_level = training_args.get_process_log_level()
    utils.LOGGING_LEVEL = log_level

    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    verify_checkpoint(logger, training_args)

    train_dataset, test_dataset = load_train_test_datasets(logger, data_args)

    config = NetfoundConfig(no_meta=data_args.no_meta, flat=data_args.flat)
    config.pretraining = False
    config.num_labels = data_args.num_labels
    config.problem_type = data_args.problem_type
    testingTokenizer = NetFoundTokenizer(config=config)

    training_config = deepcopy(config)
    training_config.p = data_args.p_val
    training_config.limit_bursts = data_args.limit_bursts
    trainingTokenizer = NetFoundTokenizer(config=training_config)
    additionalFields = None

    train_dataset = train_dataset.map(trainingTokenizer, batched=True, num_proc=get_90_percent_cpu_count())
    test_dataset = test_dataset.map(testingTokenizer, batched=True, num_proc=get_90_percent_cpu_count())

    data_collator = DataCollatorForFlowClassification(config.max_burst_length)
    if model_args.model_name_or_path is not None and os.path.exists(
            model_args.model_name_or_path
    ):
        logger.warning(f"Using weights from {model_args.model_name_or_path}")
        get_model = lambda: freeze(NetfoundFinetuningModel.from_pretrained(
            model_args.model_name_or_path, config=config
        ), model_args)
    elif model_args.no_ptm:
        get_model = lambda: NetfoundNoPTM(config=config)
    else:
        get_model = lambda: freeze(NetfoundFinetuningModel(config=config), model_args)

    if training_args.deepspeed:
        # proper deepspeed init
        model, optimizers = initialize_model_with_deepspeed(logger, training_args, get_model, frozen_base=True)
    else:
        model = get_model()
        for name, param in model.base_transformer.named_parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
        if not model_args.no_ptm:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10000, gamma=0.99
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=100, gamma=0.995
            )
        optimizers = (optimizer, lr_scheduler)

    # metrics
    problem_type = data_args.problem_type
    if problem_type == "regression":
        compute_metrics = regression_metrics
    else:
        compute_metrics = lambda p: classif_metrics(p, data_args.num_labels)

    trainer = NetfoundTrainer(
        model=model,
        extraFields=additionalFields,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=testingTokenizer,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        data_collator=data_collator,
    )
    init_tbwriter(trainer)

    if training_args.do_train:
        logger.warning("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # doesn't store tokenizer
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.warning("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
