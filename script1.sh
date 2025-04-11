# python zs_inference.py --task stance --model V3 --prompt_lang ar --save_path ./zs_preds #379
# python zs_inference.py --task pos_tagging --model V3 --prompt_lang ar --save_path ./zs_preds #680
# python zs_inference.py --task hate --model V3 --prompt_lang ar --save_path ./zs_preds #1000
# python zs_inference.py --task sarcasm --model V3 --prompt_lang en --save_path ./zs_preds #2110
# python zs_inference.py --task paraphrasing --model V3 --prompt_lang en --save_path ./zs_preds #1010



python zs_inference.py --task sarcasm --model Q14B --prompt_lang ar --save_path ./zs_preds/INS --call_limit 300