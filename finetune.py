import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import random
import os
import argparse
import torch

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

from model import DeepSeek_FT_Models
from dataset import FT_Dataset
from utils import Logger

def finetune(args):
    # SEED STUFF
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    # MODEL AND TOKENIZER
    model, tokenizer = DeepSeek_FT_Models(args.model).get_model(args)

    # DATASET PREP
    dataset_helper_train = FT_Dataset(tokenizer.eos_token, split="train", logger=logger)
    dataset_train = dataset_helper_train.get_dataset(args.task, args.prompt_lang)
    dataset_size_train = dataset_helper_train.get_size()

    # TRAIN
    max_steps = min(int((args.epochs * dataset_size_train)/(args.batch_size * args.gradient_accumulation_steps)), args.max_steps)
    
    logger("======================================================")
    logger("STARTING TRAINING")
    logger(f"Dataset size: {dataset_size_train}")
    logger(f"Max Steps: {max_steps}")
    logger(f"Total possible steps: {int((args.epochs * dataset_size_train)/(args.batch_size * args.gradient_accumulation_steps))}")
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
            lr_scheduler_type="linear",
            seed=seed,
            output_dir="outputs",
        ),
    )
    
    trainer_stats = trainer.train()
    logger("\n\n======================================================")
    logger("TRAINING FINISHED")
    logger(f"Training loss: {trainer_stats.training_loss}")
    # logger(f"Best model checkpoint: {trainer_stats.best_model_checkpoint}")

    # SAVE MODEL
    if not os.path.exists("./models"): os.mkdir("./models")
    model_path = os.path.join("./models/", f"{args.model}_{args.task}_{args.prompt_lang}")
    os.mkdir(model_path)

    model.save_pretrained(model_path) 
    tokenizer.save_pretrained(model_path)
    model.save_pretrained_merged(model_path, tokenizer, save_method = "merged_16bit")


    # # TEST
    # FastLanguageModel.for_inference(model)
    # questions = ["هذا المطعم سيء جدا", "خدمة الفندق غير جيدة", "خدمة المطعم ممتازة"]
    # for q in questions:
    #     inputs = tokenizer([dataset_helper_train.prompt_template.format(q, "")], return_tensors="pt").to("cuda")
    #     outputs = model.generate(
    #         input_ids=inputs.input_ids,
    #         attention_mask=inputs.attention_mask,
    #         max_new_tokens=1200,
    #         use_cache=True,
    #     )
    #     response = tokenizer.batch_decode(outputs)
    #     print(response[0].split(":إجابة###")[-1].replace(tokenizer.eos_token, ""))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',dest='model', default='Q1.5B', help='L8B, L70B, Q1.5B, Q7B, Q14B, Q32B')
    parser.add_argument('--prompt_lang',dest='prompt_lang', default='ar', help='ar, en')
    parser.add_argument('--task',dest='task', default='sentiment')
    parser.add_argument('--rank',dest='rank', default='4', help='4, 8, 16')
    parser.add_argument('--load_4bit',dest='load_4bit', default='0')
    parser.add_argument('--max_seq_length', dest='max_seq_length', default='2048')
    parser.add_argument('--batch_size', dest='batch_size', default='2')
    parser.add_argument('--gradient_accumulation_steps', dest='gradient_accumulation_steps', default='2')
    parser.add_argument('--epochs', dest='epochs', default='1')
    parser.add_argument('--max_steps', dest='max_steps', default='50')
    args=parser.parse_args()

    args.rank = int(args.rank)
    args.load_4bit = int(args.load_4bit)
    args.max_seq_length = int(args.max_seq_length)
    args.batch_size = int(args.batch_size)
    args.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    args.epochs = int(args.epochs)
    args.max_steps = int(args.max_steps)

    assert args.model in ["L8B", "L70B", "Q1.5B", "Q7B", "Q14B", "Q32B"], "Invalid model!"
    assert args.task in ["sentiment", "diacratization", "mcq", "pos_tagging", "summarization", "translation", "paraphrasing", "transliteration", "GQA"], "Invalid Task!"
    assert args.prompt_lang in ["en", "ar"], "Only 'en' and 'ar' languages supported!"
    assert args.rank in [4, 8, 16], "Invalid Rank!"
    assert args.load_4bit in [0, 1], "Invalid Rank!"
    assert args.max_seq_length in [512, 1024, 2048], "Invalid Rank!"
    assert args.batch_size in [2, 4, 8], "Invalid Batch Size!"
    assert args.gradient_accumulation_steps in [2], "Invalid Grad Accumulation Steps!"
    assert args.epochs > 0, "Number of epochs should be greater than 0"

    logger = Logger(os.path.join("./logs/", f"{args.model}_{args.task}_{args.prompt_lang}.txt"))
    logger("CONFIGS:")
    for arg, value in vars(args).items():
        logger(f"{arg.upper()}: {value}")
    logger("\n\n======================================================")

    try:
        finetune(args)
    except Exception as e:
        logger(e)
        logger("\n\n")
        