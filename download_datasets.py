from huggingface_hub import login
from datasets import load_dataset
 
login(token="token")

dataset_names = {
    "sentiment_train":"ajgt_twitter_ar",
    "sentiment_test":"ajgt_twitter_ar",

    # "diacratization_train":"arbml/tashkeelav2",
    # "diacratization_test":"arbml/tashkeelav2",

    # "mcq_train":"aishaalansari/CIDAR100",
    # "mcq_test":"aishaalansari/CIDAR100",

    # "pos_tagging_train":"universal_dependencies",
    # "pos_tagging_test":"universal_dependencies",

    # "summarization_train":"arbml/easc",
    # "summarization_test":"arbml/easc",

    # "translation_train":"aishaalansari/translation",
    # "translation_test":"aishaalansari/translation",

    # "paraphrasing_train": "aishaalansari/paraphrase",

    # "transliteration_train": "aishaalansari/Transliteration_NEW",
    # "transliteration_test": "aishaalansari/Transliteration_NEW",

    # "GQA_train": "asas-ai/tydiqa-goldp-ar",
    # "GQA_test": "asas-ai/tydiqa-goldp-ar",
}

dataset_splits = {
    "sentiment_train":"train[:1440]",
    "sentiment_test":"train[1440:]",

    "diacratization_train":"train",
    "diacratization_test":"test",

    "mcq_train":"train",
    "mcq_test":"test",

    "pos_tagging_train":"train",
    "pos_tagging_test":"test",

    "summarization_train":"train",
    "summarization_test":"train",

    "translation_train":"train",
    "translation_test":"test",

    "paraphrasing_train": "train",
    "paraphrasing_test": "train",

    "transliteration_train": "train",
    "transliteration_test": "test",

    "GQA_train": "train",
    "GQA_test": "validation",
}

subset_names = {
    "sentiment_train": None,
    "sentiment_test": None,

    "diacratization_train": None,
    "diacratization_test": None,

    "mcq_train": None,
    "mcq_test": None,

    "pos_tagging_train": "ar_padt",
    "pos_tagging_test": "ar_padt",

    "summarization_train": None,
    "summarization_test": None,

    "translation_train": None,
    "translation_test": None,

    "paraphrasing_train": None,
    "paraphrasing_test": None,

    "transliteration_train": None,
    "transliteration_test": None,

    "GQA_train": None,
    "GQA_test": None,
}

for key in dataset_names.keys():
    print(key)
    dataset = load_dataset(dataset_names[key], subset_names[key], split=dataset_splits[key])
    print(set(dataset["label"]))
    print()