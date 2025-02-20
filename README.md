# DeepSeek Evals

This repository is designed to finetune and evaluate the DeepSeek distilled models on Arabic NLP tasks.

<img src="/fig/ds.png">

## Requirements
```
pip install torch torchvision
pip install unsloth
pip install conllu
pip install datasets
pip install tqdm
pip install transformers
```

## Data-Prep
Please review the dataset.py file

## Fine-tuning
To fine-tune the distilled models on the Arabic NLP tasks, you can use the following command:
```
python finetune.py --model Q1.5B --task sentiment --prompt_lang ar --rank 4 --load_4bit 0 --max_seq_length 2048 --batch_size 2 --gradient_accumulation_steps 2 --epochs 1
```

- <b>model</b><br>
-- options: ['L8B', 'L70B', 'Q1.5B', 'Q7B', 'Q14B', 'Q32B']<br>
-- description: The model to be fine-tuned.<br>

- <b>task</b><br>
-- options: ['sentiment', 'diacratization', 'mcq', 'pos_tagging', 'summarization', 'translation', 'transliteration']<br>
-- description: Which NLP task to train on.<br>

- <b>prompt_lang</b><br>
-- options: ['ar', 'en']<br>
-- description: The language of the prompt.<br>

- <b>rank</b><br>
-- options: ['4', '8', '16']<br>
-- description: The rank of LoRA decomposition.<br>

- <b>load_4bit</b><br>
-- options: ['0', '1']<br>
-- description: Whether to load the 4-bit model or not.<br>

- <b>max_seq_length</b><br>
-- options: [512, 1024, 2048]<br>
-- description: The maximum sequence length.<br>

- <b>batch_size</b><br>
-- options: [2, 4, 8, 16, 32]<br>
-- description: The batch size.<br>

- <b>gradient_accumulation_steps</b><br>
-- options: [1, 2, 4, 8]<br>
-- description: The number of gradient accumulation steps.<br>

- <b>epochs</b><br>
-- description: The number of epochs.<br>

## Inference
```
python generate.py --model Q1.5B --task sentiment --prompt_lang ar
```

- <b>model</b><br>
-- options: ['L8B', 'L70B', 'Q1.5B', 'Q7B', 'Q14B', 'Q32B']<br>
-- description: The model to be fine-tuned.<br>

- <b>task</b><br>
-- options: ['sentiment', 'diacratization', 'mcq', 'pos_tagging', 'summarization', 'translation', 'transliteration']<br>
-- description: Which NLP task to train on.<br>

- <b>prompt_lang</b><br>
-- options: ['ar', 'en']<br>
-- description: The language of the prompt.<br>

## Evaluation

```
python eval.py --model Q1.5B --task sentiment --prompt_lang ar
```

- <b>model</b><br>
-- options: ['L8B', 'L70B', 'Q1.5B', 'Q7B', 'Q14B', 'Q32B']<br>
-- description: The model to be fine-tuned.<br>

- <b>task</b><br>
-- options: ['sentiment', 'diacratization', 'mcq', 'pos_tagging', 'summarization', 'translation', 'transliteration']<br>
-- description: Which NLP task to train on.<br>

- <b>prompt_lang</b><br>
-- options: ['ar', 'en']<br>
-- description: The language of the prompt.<br>