CUDA_VISIBLE_DEVICES=0 python finetune.py --task claim --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task claim --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task claim --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_claim_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task dialect --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task dialect --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task dialect --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_dialect_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task wsd --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task wsd --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task wsd --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_wsd_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task sqs --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task sqs --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task sqs --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_sqs_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task sentiment --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task sentiment --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task sentiment --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_sentiment_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task stance --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task stance --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task stance --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_stance_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task hate --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task hate --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task hate --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_hate_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task offensive --model Q14B --prompt_lang en
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task offensive --model Q14B --prompt_lang en
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task offensive --model Q14B --prompt_lang en
rm -r ./ft_models/Q14B_offensive_en

CUDA_VISIBLE_DEVICES=0 python finetune.py --task pos_tagging --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task pos_tagging --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task pos_tagging --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_pos_tagging_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task translation --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task translation --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task translation --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_translation_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task transliteration --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task transliteration --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task transliteration --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_transliteration_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task paraphrasing --model Q14B --prompt_lang en
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task paraphrasing --model Q14B --prompt_lang en
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task paraphrasing --model Q14B --prompt_lang en
rm -r ./ft_models/Q14B_paraphrasing_en

CUDA_VISIBLE_DEVICES=0 python finetune.py --task GQA --model Q14B --prompt_lang en
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task GQA --model Q14B --prompt_lang en
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task GQA --model Q14B --prompt_lang en
rm -r ./ft_models/Q14B_GQA_en

CUDA_VISIBLE_DEVICES=0 python finetune.py --task sarcasm --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task sarcasm --model Q14B --prompt_lang ar
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task sarcasm --model Q14B --prompt_lang ar
rm -r ./ft_models/Q14B_sarcasm_ar

CUDA_VISIBLE_DEVICES=0 python finetune.py --task summarization --model Q14B --prompt_lang en
CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task summarization --model Q14B --prompt_lang en
CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task summarization --model Q14B --prompt_lang en
rm -r ./ft_models/Q14B_summarization_en



