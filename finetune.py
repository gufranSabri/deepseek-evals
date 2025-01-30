import warnings
import os
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import random
import os
import argparse
import datetime
from datasets import load_dataset

from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

from datasets import load_dataset

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import Logger, FT_Dataset, DeepSeek_FT_Models

def finetune(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    model, tokenizer = DeepSeek_FT_Models(args.model).get_model(args)

    dataset_helper_train = FT_Dataset(tokenizer.eos_token, split="train")
    dataset_train = dataset_helper_train.get_dataset(args.task)
    dataset_size_train = dataset_helper_train.get_size()

    max_steps = int((args.epochs * dataset_size_train)/(args.batch_size * args.gradient_accumulation_steps))
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=seed,
            output_dir="outputs",
        ),
    )
    trainer_stats = trainer.train()

    if not os.path.exists("./models"):
        os.mkdir("./models")

    model_path = os.path.join("./models/", f"{args.model}_{args.task}")
    model.save_pretrained(model_path) 
    tokenizer.save_pretrained(model_path)
    model.save_pretrained_merged(model_path, tokenizer, save_method = "merged_16bit")


    question = "هذا المطعم سيء جدا"

    FastLanguageModel.for_inference(model)
    inputs = tokenizer([dataset_helper_train.inference_prompt_template.format(question, "")], return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    print(response[0].split("### Response:")[1])



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',dest='model', default='Q1.5B', help='L8B, L70B, Q1.5B, Q7B, Q14B, Q32B')
    parser.add_argument('--task',dest='task', default='classification')
    parser.add_argument('--rank',dest='rank', default='4', help='4, 8, 16')
    parser.add_argument('--load_4bit',dest='load_4bit', default='0')
    parser.add_argument('--max_seq_length', dest='max_seq_length', default='2048')
    parser.add_argument('--batch_size', dest='batch_size', default='2')
    parser.add_argument('--gradient_accumulation_steps', dest='gradient_accumulation_steps', default='2')
    parser.add_argument('--epochs', dest='epochs', default='1')
    args=parser.parse_args()

    args.rank = int(args.rank)
    args.load_4bit = int(args.load_4bit)
    args.max_seq_length = int(args.max_seq_length)
    args.batch_size = int(args.batch_size)
    args.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    args.epochs = int(args.epochs)

    assert args.task in ["classification", "diacratization", "mcq", "pos_tagging", "rating", "summarization", "translation"], "Invalid Task!"
    assert args.rank in [4, 8, 16], "Invalid Rank!"
    assert args.load_4bit in [0, 1], "Invalid Rank!"
    assert args.max_seq_length in [512, 1024, 2048], "Invalid Rank!"
    assert args.batch_size in [2, 4, 8], "Invalid Batch Size!"
    assert args.gradient_accumulation_steps in [2], "Invalid Grad Accumulation Steps!"
    assert args.epochs > 0, "Number of epochs should be greater than 0"

    finetune(args)