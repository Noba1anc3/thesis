from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import random
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
)

__dir__ = os.path.dirname(os.path.abspath(__file__))
__back_dir__ = os.path.join(__dir__, '../..')
sys.path.append(os.path.abspath(__dir__))
sys.path.append(os.path.abspath(__back_dir__))

from layoutlm.data.funsd import FunsdDataset
from layoutlm.modeling.layoutlm import LayoutlmConfig, LayoutlmForTokenClassification

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, LayoutlmConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForTokenClassification, BertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(data):
    batch = [i for i in zip(*data)]
    for i in range(len(batch)):
        if i < len(batch) - 2:
            batch[i] = torch.stack(batch[i], 0)
    return tuple(batch)

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=None,
    )

    # Eval!
    print("***** Running evaluation %s *****", prefix)
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)

    preds = None
    out_label_ids = None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "labels": batch[3].to(args.device),
            }
            if args.model_type in ["layoutlm"]:
                inputs["bbox"] = batch[4].to(args.device)
            inputs["token_type_ids"] = (
                batch[2].to(args.device)
                if args.model_type in ["bert", "layoutlm"]
                else None
            )  # RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            _, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list

def main():  # noqa C901
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default='data',
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default="layoutlm",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default='models/layoutlm',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='models/layoutlm',
        type=str,
    )
    ## Other parameters
    parser.add_argument(
        "--labels",
        default="data/labels.txt",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise ValueError(
            "Output directory ({}) does not exist. Please train and save the model before inference stage.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    
    args.device = device

    # Set seed
    set_seed(args)

    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )
    model = model_class.from_pretrained(args.output_dir)
    model.to(args.device)

    predictions = evaluate(
        args, model, tokenizer, labels, pad_token_label_id, mode="test")

    output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
    with open(output_test_predictions_file, "w", encoding="utf8") as writer:
        with open(
            os.path.join(args.data_dir, "test.txt"), "r", encoding="utf8"
        ) as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = (
                        line.split()[0]
                        + " "
                        + predictions[example_id].pop(0)
                        + "\n"
                    )
                    writer.write(output_line)
                else:
                    print(
                        "Maximum sequence length exceeded: No prediction for '%s'.",
                        line.split()[0],
                    )

    return predictions


if __name__ == "__main__":
    main()
