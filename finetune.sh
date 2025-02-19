CUDA_VISIBLE_DEVICES=4 python finetune.py --task paraphrasing --model R1-Q1.5B --prompt_lang en
CUDA_VISIBLE_DEVICES=4 python generate.py --task paraphrasing --model R1-Q1.5B --prompt_lang en
CUDA_VISIBLE_DEVICES=4 python eval.py --task paraphrasing --model R1-Q1.5B --prompt_lang en