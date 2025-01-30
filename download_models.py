import warnings
import os

from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import *

deepseek_models = {
    "R1":"unsloth/DeepSeek-R1",
    "L8B":"unsloth/DeepSeek-R1-Distill-Llama-8B",
    "L70B":"unsloth/DeepSeek-R1-Distill-Llama-70B",
    "Q1.5B":"unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "Q7B":"unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "Q14B":"unsloth/DeepSeek-R1-Distill-Qwen-14B",
    "Q32B":"unsloth/DeepSeek-R1-Distill-Qwen-32B",
}

for key in deepseek_models.keys():
    try:
        model_pt, tokenizer = FastLanguageModel.from_pretrained(
            model_name = deepseek_models[key],
            trust_remote_code = True
        )
    except Exception as e:
        print(e)
        print()