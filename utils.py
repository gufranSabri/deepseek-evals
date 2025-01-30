
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

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from sklearn.model_selection import train_test_split

class Logger:
    def __init__(self, file_path):
        if not os.path.exists("/".join(file_path.split("/")[:-1])):
            os.mkdir("/".join(file_path.split("/")[:-1]))
            
        self.file_path = file_path

    def __call__(self, message):
        with open(self.file_path, "a") as f:
            f.write(f"{message}\n")
            print(message)

class FT_Dataset:
    def __init__(self, EOS_TOKEN, split="train"):
        self.EOS_TOKEN = EOS_TOKEN
        self.split = split

        self.dataset_names = {
            "classification":"ajgt_twitter_ar",
            "diacratization":"arbml/tashkeelav2",
            "mcq":"arbml/cidar-mcq-100",
            "pos_tagging":"universal_dependencies",
            "rating": "arbml/cidar_alpag_chat",
            "summarization":"arbml/easc",
            "translation":"Helsinki-NLP/tatoeba_mt",
        }

        self.subset_names = {
            "classification": None,
            "diacratization": None,
            "mcq": None,
            "pos_tagging": "ar_padt",
            "rating": None,
            "summarization": None,
            "translation": "ara-rus",
        }

        self.train_prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
{}
"""

        self.inference_prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
{}
"""

        self.train_prompt_template_cot = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
{}
"""

        self.inference_prompt_template_cot = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>{}
"""

        # self.column_names = {
        #     "classification": {
        #         "X": ["text"],
        #         "Y": ["label"],
        #     },
        #     "diacratization": {
        #         "X": ["text"],
        #         "Y": ["diacratized"],
        #     },
        #     "mcq": {
        #         "X": ["Question", "A", "B", "C", "D"],
        #         "Y": ["answer"],
        #     },
        #     "pos_tagging": {
        #         "X": ["text"],
        #         "Y": ["label"],
        #     },
        #     "rating": {
        #         "X": ["text"],
        #         "Y": ["label"],
        #     },
        #     "summarization": {
        #         "X": ["article"],
        #         "Y": ["summary"],
        #     },
        #     "translation": {
        #         "X": ["sourceString"],
        #         "Y": ["targetString"],
        #     },
        # }

        self.prompt_func_map = {
            "classification": self.format_prompt_classification,
            "diacratization": self.format_prompt_diacratization,
            "mcq": self.format_prompt_mcq,
            "pos_tagging": self.format_prompt_postagging,
            "rating": None,
            "summarization": self.format_prompt_summarization,
            "translation": self.format_prompt_translation,
        }
        
        self.size = -1

    def get_size(self):
        assert self.size > 0, "Call get_dataset first() !!!"
        return self.size

    def format_prompt_classification(self, data):
        inputs = data["text"]
        outputs = data["label"]
        texts = []

        for text, label in zip(inputs, outputs):
            text = self.train_prompt_template.format(text, label) + self.EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    def format_prompt_diacratization(self, data):
        inputs = data["text"]
        outputs = data["diacratized"]
        texts = []

        for text, diacratized in zip(inputs, outputs):
            text = self.train_prompt_template.format(text, diacratized) + self.EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    def format_prompt_mcq(self, data):
        question = data["Question"]
        A, B, C, D = data["A"], data["B"], data["C"], data["D"]
        answers = data["answer"]
        texts = []

        for question, a, b, c, d, answer in zip(question, A, B, C, D, answers):
            text = self.train_prompt_template.format(question+"\n"+a+"\n"+b+"\n"+c+"\n"+d, answer) + self.EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }


    def format_prompt_postagging(self, data):
        pos_tag_classes = [ "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", "PROPN", "PRON", "X", "ADV", "INTJ", "VERB", "AUX"]

        tokenized_sents = data["tokens"]
        tags = data["upos"]
        texts = []

        outputs = []
        for i in range(len(tokenized_sents)):
            tokens = tokenized_sents[i]
            pos_tags = tags[i]

            output = ""
            for j in range(len(tokens)):
                output += tokens[j]+":"+pos_tag_classes[pos_tags[j]-1]+"\n"

            outputs.append(output)
            tokenized_sents[i] = " ".join(tokenized_sents[i])

        for inp, output in zip(tokenized_sents, outputs):
            text = self.train_prompt_template.format(inp, output) + self.EOS_TOKEN
            texts.append(text)

        return {
            "text": texts,
        }


    def format_prompt_summarization(self, data):
        articles = data["article"]
        summaries = data["summary"]
        texts = []

        for article, summary in zip(articles, summaries):
            text = self.train_prompt_template.format(article, summary) + self.EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    def format_prompt_translation(self, data):
        sourceStrings = data["sourceString"]
        targetStrings = data["targetString"]
        texts = []

        for sourceString, targetString in zip(sourceStrings, targetStrings):
            text = self.train_prompt_template.format(sourceString, targetString) + self.EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }


    def get_dataset(self, task):
        dataset_name = self.dataset_names[task]
        subset_name = self.subset_names[task]
        dataset = load_dataset(dataset_name, subset_name, split=self.split, trust_remote_code=True)

        self.size = dataset.num_rows
        dataset = dataset.map(self.prompt_func_map[task], batched = True)

        print(dataset["text"][-5])

        return dataset


class DeepSeek_FT_Models:
    def __init__(self, model_spec):
        self.model_spec = model_spec

        self.deepseek_models = {
            "R1":"unsloth/DeepSeek-R1",
            "L8B":"unsloth/DeepSeek-R1-Distill-Llama-8B",
            "L70B":"unsloth/DeepSeek-R1-Distill-Llama-70B",
            "Q1.5B":"unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
            "Q7B":"unsloth/DeepSeek-R1-Distill-Qwen-7B",
            "Q14B":"unsloth/DeepSeek-R1-Distill-Qwen-14B",
            "Q32B":"unsloth/DeepSeek-R1-Distill-Qwen-32B",
        }


    def get_model(self, args):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.deepseek_models[args.model],
            max_seq_length = args.max_seq_length,
            load_in_4bit = args.load_4bit == 1,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = 4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 42,
            use_rslora = False,
            loftq_config = None,
        )

        return model, tokenizer

if __name__ == "__main__":
    # FT_Dataset("10").get_dataset("classification")
    # FT_Dataset("10").get_dataset("diacratization")
    # FT_Dataset("10").get_dataset("mcq")
    FT_Dataset("10").get_dataset("pos_tagging")
    # FT_Dataset(10).get_dataset("rating")
    # FT_Dataset("10").get_dataset("summarization")
    # FT_Dataset("10").get_dataset("translation")

    


