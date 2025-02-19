
import os
from openai import OpenAI
from dataset import FT_Dataset
from utils import *
from tqdm import tqdm
import re
import os

def calculate_accuracy(true_list, pred_list):
    if len(true_list) != len(pred_list):
        raise ValueError("The lists must have the same length.")
    correct = sum(1 for true, pred in zip(true_list, pred_list) if true == pred)
    accuracy = correct / len(true_list)
    return accuracy


dataset_helper = FT_Dataset("<｜end▁of▁sentence｜>", split="test", test_mode=True)
dataset = dataset_helper.get_dataset("sentiment", "en")
dataset_size = dataset_helper.get_size()
 

preds = []
directory = '/home/g202302610/Code/deepseek-evals/test'
files = os.listdir(directory)
txt_files = [f for f in files if f.endswith('.txt')]
txt_files.sort(key=lambda f: int(os.path.splitext(f)[0]))
 
file_contents = []
for filename in txt_files:
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
            last_part = content[-30:]
            match = re.search(r'[01](?=\D*$)', last_part)
            file_contents.append(match.group())
        except:
            file_contents.append("-")
 
labels = dataset["label"]
for i in range(len(labels)):
    labels[i] = str(labels[i])

acc = calculate_accuracy(file_contents, labels)
print("Accuracy:", acc)


for i, text in tqdm(enumerate(dataset["text"])):
    client = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {
                "role": "user",
                "content": f"""{text}"""
            }
        ],
        temperature=0
    )

    j = completion.to_dict()
    print(j)

    res = j["choices"][0]["message"]["content"]
    logger = Logger(os.path.join("./test", f"{i}.txt"))
    logger(res)

    tqdm.write(f"{str(i)} - {res}")
