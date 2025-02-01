
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unsloth import FastLanguageModel

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
