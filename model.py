
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unsloth import FastLanguageModel

class FT_Models:
    def __init__(self, model_spec, logger=None):
        self.model_spec = model_spec
        self.logger = logger

        self.models = {
            "R1-Q1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
            "R1-Q7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
            "R1": "unsloth/DeepSeek-R1",
            "R1-L8B": "unsloth/DeepSeek-R1-Distill-Llama-8B",
            "R1-Q14B": "unsloth/DeepSeek-R1-Distill-Qwen-14B",
            
            "Q2.5-0.5B": "unsloth/Qwen2.5-0.5B",
            "Q2.5-1.5B": "unsloth/Qwen2.5-1.5B",
            "Q2.5-7B": "unsloth/Qwen2.5-7B",
            

            "P4": "unsloth/Phi-4",

            "L3.2-1B": "unsloth/Llama-3.2-1B",
            "L3.2-3B": "unsloth/Llama-3.2-3B",
        }

    def get_tokenizer(self, model_name):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.models[model_name],
            max_seq_length = 2048,
            load_in_4bit = False,
        )

        return tokenizer

    def get_model(self, args):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.models[args.model],
            max_seq_length = args.max_seq_length,
            load_in_4bit = args.load_4bit == 1,
        )

        try:
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

            if self.logger is not None:
                self.logger("LoRA on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj\n\n")
        except:
            model = FastLanguageModel.get_peft_model(
                model,
                r = 4,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_alpha = 16,
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 42,
                use_rslora = False,
                loftq_config = None,
            )

            self.logger("LoRA on q_proj, k_proj, v_proj\n\n")

        return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length = 2048,
        load_in_4bit = False,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B",
        max_seq_length = 2048,
        load_in_4bit = False,
    )
#     FT_Models("R1-Q7B")
#     FT_Models("R1")
#     FT_Models("R1-L8B")
#     FT_Models("Q2.5-0.5B")
#     FT_Models("Q2.5-1.5B")
#     FT_Models("Q2.5-7B")
#     FT_Models("P4")
#     FT_Models("L3.2-1B")
#     FT_Models("L3.2-3B")