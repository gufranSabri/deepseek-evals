import re
import os
import shutil
import argparse
from openai import OpenAI
from tqdm import tqdm

from dataset import FT_Dataset
# from model import FT_Models
from utils import *

from together import Together
import time


class ZS_Inference:
    def __init__(self, args):
        self.task = args.task
        self.model_name = args.model
        self.prompt_lang = args.prompt_lang
        self.local = True
        self.args = args
        self.shots = args.shots
        self.save_path = args.save_path
        self.call_limit = args.call_limit
        self.stop_point = args.stop_point
        self.base_url = args.base_url
        self.client = args.client
        self.api_key = args.api_key
        self.request_minute = args.request_minute
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.stream = args.stream
        self.repetition_penalty = args.repetition_penalty
        self.stop_word = args.stop_word

        self.API_MODELS = {
            # "V3": "deepseek-ai/DeepSeek-V3", #nebius
            "V3": "deepseek/deepseek_v3", #novita
            # "R1": "deepseek-ai/DeepSeek-R1", #novita
            "R1": "deepseek-reasoner",
            "Q1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            # "Q14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", #together
            "Q14B": "deepseek/deepseek-r1-distill-qwen-14b" #novita
            "GPT3.5-Turbo": "gpt-3.5-turbo",
            "GPT4": "gpt-4",
            "O1": "o1",
            "O1-Pro": "o1-pro",
            "O3-Mini": "o3_mini",
            "QWEN7B": "qwen-7b",
            "QWEN14B": "qwen-14b"
        }
        self.reasoners = ["R1", "o1", "o1-pro", "o3-mini"]
        self.shuffle = self.model_name in ["Q14B", "Q1.5B"]
        self.split = "test" if self.model_name in ["V3", "R1"] else "train"

        self.TOGETHER_MODELS = ["Q1.5B"]
        self.NOVITA_MODELS = ["V3", "Q14B"]
        self.DEEPSEEK_MODELS = ["R1"]
        self.CHATGPT_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "o1", "o1-pro", "o3-mini"]
        self.QWEN_MODELS = ["qwen-1.8b", "qwen-7b", "qwen-14b", "qwen-72b", "qwen-plus"]
        # self.NEBIUS_MODELS = []

        self.load_model()
        self.load_data()

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.preds_file_path = os.path.join(self.save_path, "_".join([self.model_name, self.task, self.prompt_lang]))
        if os.path.exists(self.preds_file_path):
            shutil.rmtree(self.preds_file_path)

        os.mkdir(self.preds_file_path)

    def load_data(self):
        self.dataset_helper = FT_Dataset("<｜end▁of▁sentence｜>", split=self.split, test_mode=False, shuffle=self.shuffle, shots=self.shots)
        self.dataset = self.dataset_helper.get_dataset(self.task, self.prompt_lang)
        self.dataset_size = self.dataset_helper.get_size()

    def load_model(self):
        if self.model_name not in self.API_MODELS.keys():
            self.load_local_model()
        else:
            self.local = False
            self.model = self.API_MODELS[self.model_name]
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

    

    def api_model_inference(self):
        print("CALLING " + self.model.upper() + " API")

        if self.client == "together":
            client = Together()
        else:
            client = OpenAI(
                base_url=os.environ.get(self.base_url),
                api_key=os.environ.get(self.api_key),
            )

            
        start_time = time.time()
        requests_sent = 0  # Track requests per minute

        for i, text in enumerate(self.dataset["text"]):
            if i == self.call_limit: break
            if i <= self.stop_point: continue

            q, a, = None, None
            if self.prompt_lang == "ar":
                q = text.split(":إجابة###")[0]
                a = text.split(":إجابة###")[1]
            else:
                q = text.split("### Response:")[0]
                a = text.split("### Response:")[1]

            if requests_sent >= self.request_minute:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60 and self.request_minute > 0:
                    time.sleep(60 - elapsed_time)  # Wait for the remaining time
                start_time = time.time()  # Reset start time
                requests_sent = 0  # Reset counter

            chat_completion_res = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""{q}""",
                    }
                ],
                max_completion_tokens= self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                # stop=[self.stop_word],
                stream=self.stream
            )

            res = chat_completion_res.choices[0].message.content

            logger = Logger(os.path.join(self.preds_file_path, f"{i}.txt"))
                
            logger(q)
            logger("=================================================================================")
            logger(a)
            logger("=================================================================================")
            logger(res)
            if self.model_name in self.reasoners:
                logger("=================================================================================")
                res_r = chat_completion_res.choices[0].message.reasoning_content
                logger(res_r)
            print(f"{i} ------------------------------------------\n\n\n")
            requests_sent += 1  # Increment request count
            time.sleep(1.2)  # Ensure ~50 requests per minute

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
                logger(response[0].split(":إجابة###")[1].replace("<｜end▁of▁sentence｜>", "").replace("<｜end▁of▁sentence｜>", ""))
            else:
                logger(response[0].split("### Response:")[1].replace("<｜end▁of▁sentence｜>", "").replace("<｜end▁of▁sentence｜>", ""))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',dest='model', default='R1-Q1.5B')
    parser.add_argument('--prompt_lang',dest='prompt_lang', default='ar', help='ar, en')
    parser.add_argument('--task',dest='task', default='paraphrasing')
    parser.add_argument('--rank',dest='rank', default='4', help='4, 8, 16')
    parser.add_argument('--load_4bit',dest='load_4bit', default='0')
    parser.add_argument('--max_seq_length', dest='max_seq_length', default='2048')
    parser.add_argument('--batch_size', dest='batch_size', default='2')
    parser.add_argument('--shots', dest='shots', default='0')
    parser.add_argument('--save_path', dest='save_path', default='./zs_preds')
    parser.add_argument('--call_limit', dest='call_limit', default="10000")
    parser.add_argument('--stop_point', dest='stop_point', default="-1")
    parser.add_argument('--request_minute', dest='request_minute', default="-1")
    parser.add_argument('--base_url', dest='base_url', default="")
    parser.add_argument('--api_key', dest='api_key', default="OPENAI_API")
    parser.add_argument('--client', dest='client', default="OpenAI")
    parser.add_argument('--temperature', dest='temperature', default=None)
    parser.add_argument('--top_p', dest='top_p', default=None)
    parser.add_argument('--top_k', dest='top_k', default=None)
    parser.add_argument('--max_tokens', dest='max_tokens', default=None)
    parser.add_argument('--repetition_penalty', dest='repetition_penalty', default=None)
    parser.add_argument('--stop_word', dest='stop_word', default=None)
    parser.add_argument('--stream', dest='stream', default=None)

    args=parser.parse_args()

    args.rank = int(args.rank)
    args.load_4bit = int(args.load_4bit)
    args.max_seq_length = int(args.max_seq_length)
    args.batch_size = int(args.batch_size)
    args.shots = int(args.shots)
    args.call_limit = int(args.call_limit)
    args.stop_point = int(args.stop_point)
    args.request_minute = int(args.request_minute)
    args.temperature = float(args.temperature) if args.temperature is not None else args.temperature
    args.top_p = int(args.top_p) if args.top_p is not None else args.top_p
    args.top_k = int(args.top_k) if args.top_k is not None else args.top_k
    args.repetition_penalty = int(args.repetition_penalty) if args.repetition_penalty is not None else args.repetition_penalty
    args.stream = bool(args.stream) if args.stream is not None else args.stream


    assert args.model in ["V3", "R1", "Q1.5B", "Q14B"], "Invalid model!"
    assert args.prompt_lang in ["en", "ar"], "Only 'en' and 'ar' languages supported!"
    assert args.rank in [4, 8, 16], "Invalid Rank!"
    assert args.load_4bit in [0, 1], "Invalid Rank!"
    assert args.max_seq_length in [512, 1024, 2048], "Invalid Sequence Length!"
    assert args.batch_size in [2, 4, 8], "Invalid Batch Size!"

    zs = ZS_Inference(args)
    zs.inference()
