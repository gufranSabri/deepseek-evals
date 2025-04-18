
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
            "Q1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
            "Q7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
            "Q14B": "unsloth/DeepSeek-R1-Distill-Qwen-14B",
        }

    def get_tokenizer(self, model_name):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.models[model_name],
            max_seq_length = 1024,
            load_in_4bit = False,
        )

        return tokenizer

    def get_zs_model(self, args):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.models[args.model],
            max_seq_length = args.max_seq_length,
            load_in_4bit = args.load_4bit == 1,
        )
        FastLanguageModel.for_inference(model)

        return model, tokenizer

    def get_ft_model(self, args):
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