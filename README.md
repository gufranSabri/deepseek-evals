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
Download data from <a href="https://kfupmedusa-my.sharepoint.com/:u:/g/personal/g202302610_kfupm_edu_sa/EfdWKQ54vFpMtNvzCJDcbbMBd9ko5-iuXUeQQXkwcARE1A?e=wlpl71">here</a> and place in `./data` folder

For more information please review the `dataset.py` file

## Scripts
Please review `scripts` folder for zero shot, few shot, fine-tuning scripts.