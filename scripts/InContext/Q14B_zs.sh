python zs_inference.py --task stance --model Q14B --prompt_lang ar --save_path ./zs_preds #379
python zs_inference.py --task pos_tagging --model Q14B --prompt_lang ar --save_path ./zs_preds #680
python zs_inference.py --task hate --model Q14B --prompt_lang ar --save_path ./zs_preds #1000
python zs_inference.py --task sarcasm --model Q14B --prompt_lang ar --save_path ./zs_preds #2110
python zs_inference.py --task paraphrasing --model Q14B --prompt_lang en --save_path ./zs_preds #1010
python zs_inference.py --task claim --model Q14B --prompt_lang ar --save_path ./zs_preds #456
python zs_inference.py --task GQA --model Q14B --prompt_lang en --save_path ./zs_preds #921
python zs_inference.py --task offensive --model Q14B --prompt_lang en --save_path ./zs_preds #1000
python zs_inference.py --task dialect --model Q14B --prompt_lang ar --save_path ./zs_preds #2110
python zs_inference.py --task sqs --model Q14B --prompt_lang ar --save_path ./zs_preds #3715
python zs_inference.py --task wsd --model Q14B --prompt_lang ar --save_path ./zs_preds #6220
python zs_inference.py --task summarization --model Q14B --prompt_lang en --save_path ./zs_preds #6220
python zs_inference.py --task translation --model Q14B --prompt_lang ar --save_path ./zs_preds #6220
python zs_inference.py --task transliteration --model Q14B --prompt_lang ar --save_path ./zs_preds #6220
python zs_inference.py --task sentiment --model Q14B --prompt_lang ar --save_path ./zs_preds #6220


python zs_eval.py --task stance --model Q14B --prompt_lang ar