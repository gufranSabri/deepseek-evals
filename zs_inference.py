import re
import os
import shutil
import argparse
from openai import OpenAI
from tqdm import tqdm
from unsloth import FastLanguageModel

from dataset import FT_Dataset
from model import FT_Models
from utils import *


class ZS_Inference:
    def __init__(self, args):
        self.task = args.task
        self.model_name = args.model
        self.prompt_lang = args.prompt_lang
        self.local = True
        self.args = args
        self.API_MODELS = {
            "V3": "deepseek-ai/DeepSeek-V3",
            "R1": "deepseek-ai/DeepSeek-R1",
        }

        self.load_model()
        self.load_data()

        if not os.path.exists("./zs_preds"):
            os.mkdir("./zs_preds")

        self.preds_file_path = os.path.join("./zs_preds", "_".join([self.model_name, self.task, self.prompt_lang]))
        if os.path.exists(self.preds_file_path):
            shutil.rmtree(self.preds_file_path)

        os.mkdir(self.preds_file_path)

    def load_data(self):
        self.dataset_helper = FT_Dataset(self.tokenizer.eos_token, split="test", test_mode=True)
        self.dataset = self.dataset_helper.get_dataset(self.task, self.prompt_lang)
        self.dataset_size = self.dataset_helper.get_size()

    def load_model(self):
        if self.model_name not in self.API_MODELS.keys():
            self.load_local_model()
        else:
            self.local = False
            self.model = self.API_MODELS[self.model_name]
            self.tokenizer = FT_Models(self.model_name).get_tokenizer("R1-Q1.5B")

            print("Will call API on", self.model)
    
    def load_local_model(self):
        model, tokenizer = FT_Models(self.model_name).get_zs_model(self.args)
        self.tokenizer = tokenizer
        self.model = model

    def inference(self):
        if self.local:
            self.local_model_inference()
        else:
            self.api_model_inference()

    def local_model_inference(self):
        for i, prompt in enumerate(self.dataset["text"]):
            print(f"{i} ------------------------------------------\n\n")
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1200,
                use_cache=True,
            )
            response = self.tokenizer.batch_decode(outputs)

            logger = Logger(os.path.join(self.preds_file_path, f"{i}.txt"))
            if self.prompt_lang == "ar":
                logger(response[0].split(":إجابة###")[1].replace(self.tokenizer.eos_token, ""))
            else:
                logger(response[0].split("### Response:")[1].replace(self.tokenizer.eos_token, ""))
            
    def api_model_inference(self):
        for i, text in enumerate(self.dataset["text"]):
            client = OpenAI(
                base_url="https://api.studio.nebius.ai/v1/",
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""{text}"""
                    }
                ],
                temperature=0
            )

            j = completion.to_dict()
            res = j["choices"][0]["message"]["content"]
            logger = Logger(os.path.join(self.preds_file_path, f"{i}.txt"))
            logger(res)
            print(f"{i} ------------------------------------------\n")


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',dest='model', default='R1-Q1.5B')
    parser.add_argument('--prompt_lang',dest='prompt_lang', default='en', help='ar, en')
    parser.add_argument('--task',dest='task', default='paraphrasing')
    parser.add_argument('--rank',dest='rank', default='4', help='4, 8, 16')
    parser.add_argument('--load_4bit',dest='load_4bit', default='0')
    parser.add_argument('--max_seq_length', dest='max_seq_length', default='2048')
    parser.add_argument('--batch_size', dest='batch_size', default='2')
    args=parser.parse_args()

    args.rank = int(args.rank)
    args.load_4bit = int(args.load_4bit)
    args.max_seq_length = int(args.max_seq_length)
    args.batch_size = int(args.batch_size)

    assert args.model in ["R1-Q1.5B", "R1-Q7B", "R1-Q14B", "V3", "R1"], "Invalid model!"
    assert args.prompt_lang in ["en", "ar"], "Only 'en' and 'ar' languages supported!"
    assert args.rank in [4, 8, 16], "Invalid Rank!"
    assert args.load_4bit in [0, 1], "Invalid Rank!"
    assert args.max_seq_length in [512, 1024, 2048], "Invalid Sequence Length!"
    assert args.batch_size in [2, 4, 8], "Invalid Batch Size!"

    zs = ZS_Inference(args)
    zs.inference()