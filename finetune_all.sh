#taqyim datasets - ordered on basis of size

CUDA_VISIBLE_DEVICES=0 python finetune.py --task sentiment
CUDA_VISIBLE_DEVICES=0 python generate.py --task sentiment
CUDA_VISIBLE_DEVICES=0 python eval.py --task sentiment

CUDA_VISIBLE_DEVICES=0 python finetune.py --task pos_tagging
CUDA_VISIBLE_DEVICES=0 python generate.py --task pos_tagging
CUDA_VISIBLE_DEVICES=0 python eval.py --task pos_tagging

CUDA_VISIBLE_DEVICES=0 python finetune.py --task paraphrasing
CUDA_VISIBLE_DEVICES=0 python generate.py --task paraphrasing
CUDA_VISIBLE_DEVICES=0 python eval.py --task paraphrasing

CUDA_VISIBLE_DEVICES=0 python finetune.py --task summarization
CUDA_VISIBLE_DEVICES=0 python generate.py --task summarization
CUDA_VISIBLE_DEVICES=0 python eval.py --task summarization

CUDA_VISIBLE_DEVICES=0 python finetune.py --task transliteration
CUDA_VISIBLE_DEVICES=0 python generate.py --task transliteration
CUDA_VISIBLE_DEVICES=0 python eval.py --task transliteration

# CUDA_VISIBLE_DEVICES=1 python finetune.py --task diacratization
# CUDA_VISIBLE_DEVICES=1 python generate.py --task diacratization
# CUDA_VISIBLE_DEVICES=1 python eval.py --task diacratization
# python finetune.py --task translation

# =================================
# python finetune.py --task mcq
# python finetune.py --task GQA