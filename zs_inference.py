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

        self.API_MODELS = {
            # "V3": "deepseek-ai/DeepSeek-V3", #nebius
            # "V3": "deepseek/deepseek_v3", #novita
            "V3" : "deepseek-chat", #deepseek

            # "R1": "deepseek-ai/DeepSeek-R1",
            "R1": "deepseek-reasoner",

            "QWQ": "qwen/qwq-32b",
            
            "Q1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", #nebius
            "Q14B": "deepseek/deepseek-r1-distill-qwen-14b", #novita
            "Q32B": "deepseek/deepseek-r1-distill-qwen-32b" #novita
        }
        self.shuffle = self.model_name in ["Q14B", "Q1.5B"]
        self.split = "test" if self.model_name in ["V3", "R1", "QWQ", "Q14B", "Q32B"] else "train"
        args.resume = True if args.resume == "1" else False

        print("SPLIT:", self.split)

        self.TOGETHER_MODELS = ["Q1.5B"]
        self.NOVITA_MODELS = ["Q14B", "R1", "QWQ", "Q32B"]
        self.DEEPSEEK_MODELS = ["R1", "V3"]

        self.load_model()
        self.load_data()

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.preds_file_path = os.path.join(self.save_path, "_".join([self.model_name, self.task, self.prompt_lang]))
        self.start = 0
        if os.path.exists(self.preds_file_path) and not args.resume:
            print("Deleting previous predictions...")
            shutil.rmtree(self.preds_file_path)
            self.start = len(os.listdir(self.preds_file_path)) if os.path.exists(self.preds_file_path) else 0
            os.mkdir(self.preds_file_path)
        elif not os.path.exists(self.preds_file_path):
            os.mkdir(self.preds_file_path)
            self.start = 0
        elif args.resume:
            print("Resuming from previous predictions...")
            self.start = len(os.listdir(self.preds_file_path))

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
            if self.model_name in self.TOGETHER_MODELS:
                self.api_model_inference_together()
            elif self.model_name in self.NOVITA_MODELS:
                self.api_model_inference_novita()
            elif self.model_name in self.DEEPSEEK_MODELS:
                self.api_model_inference_deepseek()

    def api_model_inference_deepseek(self):
        print("CALLING DEEPSEEK API")

        for i, text in enumerate(self.dataset["text"]):
            if i == self.call_limit: break

            q, a, = None, None
            if self.prompt_lang == "ar":
                q = text.split(":إجابة###")[0]
                a = text.split(":إجابة###")[1]
            else:
                q = text.split("### Response:")[0]
                a = text.split("### Response:")[1]

            client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key="sk-1023ce7c74fa4c5aaadd299c2758f0e3",
            ) 


            chat_completion_res = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""{q}""",
                    }
                ],
                # max_tokens=1024,
            )

            res = chat_completion_res.choices[0].message.content
            logger = Logger(os.path.join(self.preds_file_path, f"{i}.txt"))
                
            logger(q)
            logger("=================================================================================")
            logger(a)
            logger("=================================================================================")
            logger(res)
            print(f"{i} ------------------------------------------\n\n\n")

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

    def api_model_inference_novita(self):
        print("CALLING NOVITA API")

        for i, text in enumerate(self.dataset["text"]):
            if i < self.start: continue
            if i == self.call_limit: break

            q, a, = None, None
            if self.prompt_lang == "ar":
                q = text.split(":إجابة###")[0]
                a = text.split(":إجابة###")[1]
            else:
                q = text.split("### Response:")[0]
                a = text.split("### Response:")[1]

            client = OpenAI(
                base_url="https://api.novita.ai/v3/openai",
                api_key="sk_XTZ7jxWmpXeqCiVG-QuFN3jj1FiDLK1EyOVudLleglk",
            ) 

            if self.task == "summarization" and i < 9921:
                res = "<think>\nempty\n</think>\n\n<answer>0</answer>"

            else:
                chat_completion_res = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""{q}""",
                        }
                    ],
                    # max_tokens=1024,
                )
                res = chat_completion_res.choices[0].message.content

            logger = Logger(os.path.join(self.preds_file_path, f"{i}.txt"))
                
            logger(q)
            logger("=================================================================================")
            logger(a)
            logger("=================================================================================")
            logger(res)

            print(f"{i} ------------------------------------------\n\n\n")
    
    def api_model_inference_together(self):
        print("CALLING TOGETHER API")

        client = Together()
        start_time = time.time()
        requests_sent = 0  # Track requests per minute

        for i, text in enumerate(self.dataset["text"]):
            if i < self.start: continue
            if i == self.call_limit: break

            q, a = None, None
            if self.prompt_lang == "ar":
                q = text.split(":إجابة###")[0]
                a = text.split(":إجابة###")[1]
            else:
                q = text.split("### Response:")[0]
                a = text.split("### Response:")[1]

            if requests_sent >= 50:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    time.sleep(60 - elapsed_time)  # Wait for the remaining time
                start_time = time.time()  # Reset start time
                requests_sent = 0  # Reset counter

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"""{q}"""}],
                # max_tokens=1024,
                temperature=0.7,
                top_p=0.3,
                top_k=50,
                repetition_penalty=1,
                stop=["<｜end▁of▁sentence｜>"],
                stream=False
            )
            res = response.choices[0].message.content

            logger = Logger(os.path.join(self.preds_file_path, f"{i}.txt"))
                
            logger(q)
            logger("=================================================================================")
            logger(a)
            logger("=================================================================================")
            logger(res)

            requests_sent += 1  # Increment request count
            time.sleep(1.2)  # Ensure ~50 requests per minute

            print(f"{i} ------------------------------------------\n\n\n")


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
    parser.add_argument('--call_limit', dest='call_limit', default="20000")
    parser.add_argument('--resume', dest='resume', default="1")
    args=parser.parse_args()

    args.rank = int(args.rank)
    args.load_4bit = int(args.load_4bit)
    args.max_seq_length = int(args.max_seq_length)
    args.batch_size = int(args.batch_size)
    args.shots = int(args.shots)
    args.call_limit = int(args.call_limit)

    assert args.model in ["V3", "R1", "Q1.5B", "Q14B", "QWQ", "Q32B"], "Invalid model!"
    assert args.prompt_lang in ["en", "ar"], "Only 'en' and 'ar' languages supported!"
    assert args.rank in [4, 8, 16], "Invalid Rank!"
    assert args.load_4bit in [0, 1], "Invalid Rank!"
    assert args.max_seq_length in [512, 1024, 2048], "Invalid Sequence Length!"
    assert args.batch_size in [2, 4, 8], "Invalid Batch Size!"

    zs = ZS_Inference(args)
    zs.inference()
