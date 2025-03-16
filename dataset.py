import warnings
import os
import pickle
import pandas as pd
from datasets import Dataset
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import login
from datasets import load_dataset

class FT_Dataset:
    def __init__(self, EOS_TOKEN, split="train", shots=0, logger = None, test_mode=False, shuffle=False):
        login(token="token")

        assert shots in [0, 1, 3, 5, 10], "Shots should be one of 0, 1, 3, 5, 10"
        self.shots = shots

        self.EOS_TOKEN = "" if test_mode else EOS_TOKEN
        self.split = split
        self.logger = logger
        self.test_mode = test_mode

        self.shuffle = shuffle

        print("WILL SHUFFLE: " + str(self.shuffle) + " =====================================")

        self.dataset_names = {
            "sentiment_train":"ajgt_twitter_ar",
            "sentiment_test":"ajgt_twitter_ar",

            "pos_tagging_train":"universal_dependencies",
            "pos_tagging_test":"universal_dependencies",

            "summarization_train":"./data/sum_train.csv",
            "summarization_test":"./data/sum_test.csv",

            "translation_train":"./data/translation_train.csv",
            "translation_test":"./data/translation_test.csv",

            "paraphrasing_train": "aishaalansari/paraphrase" ,
            "paraphrasing_test": "aishaalansari/Paraphrasing",

            "transliteration_train": "./data/transliteration_train.csv",
            "transliteration_test": "./data/transliteration_test.csv",

            "paraphrase_detection_train": "./data/paraphrase_detection_train.csv",
            "paraphrase_detection_test": "./data/paraphrase_detection_test.csv",

            "stance_train": "./data/stance_train.csv",
            "stance_test": "./data/stance_test.csv",

            "claim_train": "./data/claim_train.csv",
            "claim_test": "./data/claim_test.csv",

            "wsd_train": "./data/wsd_train.csv",
            "wsd_test": "./data/wsd_test.csv",

            "mcq_train":"aishaalansari/CIDAR100",
            "mcq_test":"aishaalansari/CIDAR100",

            "GQA_train": "asas-ai/tydiqa-goldp-ar",
            "GQA_test": "asas-ai/tydiqa-goldp-ar",

            "diacratization_train":"arbml/tashkeelav2",
            "diacratization_test":"arbml/tashkeelav2",
        }

        self.dataset_splits = {
            "sentiment_train":"train[:1440]",
            "sentiment_test":"train[1440:]",

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

            "paraphrase_detection_train":"train",
            "paraphrase_detection_test":"test",

            "claim_train":"train",
            "claim_test":"test",

            "stance_train":"train",
            "stance_test":"test",

            "mcq_train":"train",
            "mcq_test":"test",

            "GQA_train": "train",
            "GQA_test": "validation",

            "diacratization_train":"train",
            "diacratization_test":"test",
        }

        self.subset_names = {
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

            "GQA_train": None,
            "GQA_test": None,
        }

        self.prompt_func_map = {
            "sentiment_train": self.format_prompt_sentiment,
            "sentiment_test": self.format_prompt_sentiment,

            "diacratization_train": self.format_prompt_diacratization,
            "diacratization_test": self.format_prompt_diacratization,

            "mcq_train": self.format_prompt_mcq,
            "mcq_test": self.format_prompt_mcq,

            "pos_tagging_train": self.format_prompt_postagging,
            "pos_tagging_test": self.format_prompt_postagging,

            "summarization_train": self.format_prompt_summarization,
            "summarization_test": self.format_prompt_summarization,

            "translation_train": self.format_prompt_translation,
            "translation_test": self.format_prompt_translation,

            "paraphrasing_train": self.format_prompt_paraphrasing,
            "paraphrasing_test": self.format_prompt_paraphrasing,

            "transliteration_train": self.format_prompt_transliteration,
            "transliteration_test": self.format_prompt_transliteration,

            "GQA_train": self.format_prompt_GQA,
            "GQA_test": self.format_prompt_GQA,

            "paraphrase_detection_train": self.format_prompt_paraphrase_detection,
            "paraphrase_detection_test": self.format_prompt_paraphrase_detection,

            "claim_train": self.format_prompt_claim,
            "claim_test": self.format_prompt_claim,

            "stance_train": self.format_prompt_stance,
            "stance_test": self.format_prompt_stance,

            "wsd_train": self.format_prompt_wsd,
            "wsd_test": self.format_prompt_wsd,
        }

        self.task_instructions = {
           "sentiment": "Classify the sentiment of this sentence as 0 for Negative or 1 for Positive. Give the answer in arabic.",
           "summarization": "Summarize the following text in one sentence. Give the answer in arabic.",
           "paraphrasing": "Paraphrase the following text while keeping the meaning intact. Give the answer in arabic.",
           "transliteration": "Convert the following text written in Arabizi into Arabic script. Give the answer in arabic.",
           "GQA": "Generate an informative answer for the following question. Give the answer in arabic.",
           "grammar_correction": "Correct the grammatical errors in this sentence. Give the answer in arabic.",
           "translation": "Translate the following English sentence into Arabic. Give the answer in arabic.",
        }

        # self.task_instructions = {
        #     "sentiment": "What sentiment does this text express? Type '1' for Positive and '0' for Negative. Give the answer in arabic.",
        #     "summarization": "Can you summarize the following text in one sentence? Give the answer in arabic.",
        #     "paraphrasing": "Can you paraphrase the following text while preserving the meaning? Give the answer in arabic.",
        #     "transliteration": "Can you convert the following text into Arabic script? Give the answer in arabic.",
        #     "GQA": "What is the answer for the following question? Give the answer in arabic.",
        #     "grammar_correction": "Does this sentence have any grammatical errors? If yes, provide the correction. Otherwise, re-write the sentence. Give the answer in arabic.",
        #     "translation": "What is the translation of this sentence from English to Arabic? Give the answer in arabic.",
        # }

        # self.task_instructions = {
        #     "sentiment": "You are an expert in sentiment analysis and natural language processing. Analyze the given text and answer 1 if the sentiment is Positive and 0 if the sentiment is Negative. Give the answer in arabic.",
        #     "summarization": "You are a professional text summarizer with expertise in extracting key information. Read the given text and generate a one-sentence concise and coherent summary that preserves the main ideas and important details. Give the answer in arabic.",
        #     "paraphrasing": "You are a language expert skilled in rewriting text while maintaining its original meaning. Rewrite the following passage in arabic using different words and sentence structures while keeping the meaning intact. Give the answer in arabic.",
        #     "transliteration": "You are an expert in Arabizi transliteration. Convert the following text from Arabizi into Arabic script. Give the answer in arabic.",
        #     "GQA": "You are an advanced knowledge-based AI trained in answering general questions across multiple domains. Provide an accurate, well-structured, and informative response to the following question. Give the answer in arabic.",
        #     "grammar_correction": "You are a professional proofreader. Read the following sentence and correct any grammatical mistakes. Give the answer in arabic.",
        #     "translation": "What is the translation of this sentence from English to Arabic? Give the answer in arabic.",
        # }

        self.task_instructions_ar = {
           "sentiment": "صنف مشاعر هذه الجملة كـ 0 إذا كانت سلبية و 1 إذا كانت إيجابية. قم بالاجابة باللغة العربية ",
           "summarization": "لخص النص التالي في جملة واحدة. قم بالاجابة باللغة العربية ",
           "paraphrasing": "أعد صياغة النص التالي مع الحفاظ على المعنى كما هو. قم بالاجابة باللغة العربية ",
           "transliteration": "حوّل النص التالي المكتوب بالحروف اللاتينية وفقًا لأسلوب العربيزي إلى الحروف العربية. قم بالاجابة باللغة العربية ",
           "GQA": "قم بإنشاء إجابة توضيحية للسؤال التالي. قم بالاجابة باللغة العربية ",
           "grammar_correction": "صحح الأخطاء النحوية في هذه الجملة. قم بالاجابة باللغة العربية ",
           "translation": "ترجم الجملة الإنجليزية التالية إلى اللغة العربية. قم بالاجابة باللغة العربية ",
        }

        # self.task_instructions_ar = {
        #    "sentiment": "هل تعبر هذه الجملة عن مشاعر إيجابية أم سلبية؟ اكتب '1' إذا كانت إيجابية و'0' إذا كانت سلبية. قم بالاجابة باللغة العربية ",
        #    "summarization": "هل يمكنك تلخيص النص التالي في جملة واحدة؟ قم بالاجابة باللغة العربية ",
        #    "paraphrasing": "هل يمكنك إعادة صياغة النص التالي دون تغيير معناه؟ قم بالاجابة باللغة العربية ",
        #    "transliteration": "هل يمكنك تحويل النص التالي إلى الحروف العربية؟ قم بالاجابة باللغة العربية ",
        #    "GQA": "ما هي الإجابة لهذا السؤال؟ قم بالاجابة باللغة العربية ",
        #    "grammar_correction": "هل تحتوي هذه الجملة على أخطاء نحوية؟ إذا كانت الإجابة نعم، قم بتصحيح الجملة. إذا لم تحتوِ على أخطاء، أعد كتابة الجملة. قم بالاجابة باللغة العربية ",
        #    "translation": "ماهي ترجمة هذه الجملة باللغة الانجليزية الى العربية؟. قم بالاجابة باللغة العربية ",
        # }

        # self.task_instructions_ar = {
        #     "sentiment": "أنت خبير في تحليل المشاعر ومعالجة اللغة الطبيعية. قم بتحليل النص المعطى وأجب بـ 1 إذا كانت المشاعر إيجابية و0 إذا كانت المشاعر سلبية. قم بالاجابة باللغة العربية ",
        #     "summarization": "أنت متخصص في تلخيص النصوص ولديك خبرة في استخراج المعلومات الأساسية. اقرأ النص المقدم وأنشئ ملخصًا موجزًا ومتماسكًا في جملة واحدة يحافظ على الأفكار الرئيسية والتفاصيل المهمة. قم بالاجابة باللغة العربية ",
        #     "paraphrasing": "أنت خبير لغوي ماهر في إعادة صياغة النص مع الحفاظ على معناه الأصلي. أعد كتابة المقطع التالي باستخدام كلمات وهياكل جمل مختلفة مع الحفاظ على المعنى كما هو. قم بالاجابة باللغة العربية ",
        #     "transliteration": "أنت خبير في تحويل النصوص المكتوبة بالأحرف اللاتينية وفقًا لأسلوب العربيزي. حوّل النص التالي إلى الحروف العربية. قم بالاجابة باللغة العربية ",
        #     "GQA": "أنت ذكاء اصطناعي متقدم قائم على المعرفة ومدرب على الإجابة على الأسئلة العامة عبر مجالات متعددة. قدم إجابة دقيقة، ومنظمة للسؤال التالي. قم بالاجابة باللغة العربية ",
        #     "grammar_correction": "أنت مدقق لغوي محترف. اقرأ الجملة التالية وصحح أي أخطاء نحوية. قم بالاجابة باللغة العربية ",
        #     "translation": "أنت مترجم محترف تتقن اللغتين الإنجليزية والعربية. ترجم الجملة التالية إلى اللغة العربية مع الالتزام بالقواعد اللغوية الصحيحة والحفاظ على السياق السليم. قم بالاجابة باللغة العربية ",
        # }

        self.size = -1

    def get_size(self):
        assert self.size > 0, "Call get_dataset() first !!!"
        return self.size

    def format_prompt_sentiment(self, data):
        inputs = data["text"]
        outputs = data["label"]
        texts = []
        
        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(inputs), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += inputs[i] + "\n<answer>" + str(outputs[i]) + "</answer>\n\n"
        else:
            examples = ""

        for text, label in zip(inputs, outputs):
            text = self.prompt_template.format(examples, text, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}
        

    def format_prompt_diacratization(self, data):
        inputs = data["text"]
        outputs = data["diacratized"]
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(inputs), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += inputs[i] + "\n<answer>" + outputs[i] + "</answer>\n\n"
        else:
            examples = ""

        for text, diacratized in zip(inputs, outputs):
            text = self.prompt_template.format(examples, text, diacratized if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_mcq(self, data):
        question = data["Question"]
        A, B, C, D = data["A"], data["B"], data["C"], data["D"]
        answers = data["answer"]
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(question), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += question[i] + "\n" + A[i] + "\n" + B[i] + "\n" + C[i] + "\n" + D[i] + "\n<answer>" + answers[i] + "</answer>\n\n"
        else:
            examples = ""

        for question, a, b, c, d, answer in zip(question, A, B, C, D, answers):
            text = self.prompt_template.format(examples, question+"\n"+a+"\n"+b+"\n"+c+"\n"+d, answer if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)

        return {"text": texts}


    def format_prompt_postagging(self, data):
        pos_tag_classes = [ "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", "PROPN", "PRON", "X", "ADV", "INTJ", "VERB", "AUX"]

        tokenized_sents = data["tokens"]
        tags = data["upos"]
        texts = []

        outputs = []
        for i in range(len(tokenized_sents)):
            tokens = tokenized_sents[i]
            pos_tags = tags[i]

            output = ""
            for j in range(len(tokens)):
                output += tokens[j]+":"+pos_tag_classes[pos_tags[j]-1]+"\n"

            outputs.append(output)
            tokenized_sents[i] = " ".join(tokenized_sents[i])

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(tokenized_sents), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += tokenized_sents[i] + "\n<answer>" + outputs[i] + "</answer>\n\n"
        else:
            examples = ""

        for inp, output in zip(tokenized_sents, outputs):
            text = self.prompt_template.format(examples, inp, output if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)

        return {"text": texts}


    def format_prompt_summarization(self, data):
        X_col = "article"
        y_col = "summary"

        articles = data[X_col]
        summaries = data[y_col]
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(articles), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += articles[i] + "\n<answer>" + summaries[i] + "</answer>\n\n"
        else:
            examples = ""

        for article, summary in zip(articles, summaries):
            text = self.prompt_template.format(examples, article, summary if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_translation(self, data):
        sourceStrings = data["sourceString"]
        targetStrings = data["targetString"]
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(sourceStrings), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += sourceStrings[i] + "\n<answer>" + targetStrings[i] + "</answer>\n\n"
        else:
            examples = ""

        for sourceString, targetString in zip(sourceStrings, targetStrings):
            text = self.prompt_template.format(examples, sourceString, targetString if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_paraphrasing(self, data):
        sentences, paraphrases = [], []

        if self.split == "test":
            temp = data["First sentence;second sentence;44_experts;similarity;parahrase"]
            for d in temp:
                d = d.split(";")
                sentences.append(d[0])
                paraphrases.append(d[1])
        else:
            sentences = data["Source"]
            paraphrases = data["Target"]

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(sentences), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += sentences[i] + "\n<answer>" + paraphrases[i] + "</answer>\n\n"
        else:
            examples = ""

        texts = []
        for sent, para in zip(sentences, paraphrases):
            text = self.prompt_template.format(examples, sent, para if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    
    def format_prompt_transliteration(self,data):
        EN = data["source"]
        AR = data["transliteration"]

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(EN), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += EN[i] + "\n<answer>" + AR[i] + "</answer>\n\n"
        else:
            examples = ""
        
        texts = []
        for en, ar in zip(EN, AR):
            text = self.prompt_template.format(examples, en, ar if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_GQA(self, data):
        question = data["question_text"]
        answer = data["answers"]
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(question), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += question[i] + "\n<answer>" + answer[i]["text"] + "</answer>\n\n"
        else:
            examples = ""
 
        for q, a in zip(question, answer):
            text = self.prompt_template.format(examples, q, a["text"] if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_paraphrase_detection(self, data):
        question1 = data["question1"]
        question2 = data["question2"]
        questions = zip(question1, question2)
        labels = data["label"]
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(question1), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += "سؤال ١: "+ question1[i] + "\n" + "سؤال ٢: " + question2[i] + "\n<answer>" + labels[i] + "</answer>\n\n"
        else:
            examples = ""
 
        for (question1, question2), label in zip(questions, labels):
            q_res = "سؤال ١: "+ question1 + "\n" + "سؤال ٢: " + question2
            text = self.prompt_template.format(examples, q_res, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_claim(self, data):
        claims = data["claim_s"]
        flags = data["fake_flag"]
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(claims), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += claims[i] + "\n<answer>" + flags[i] + "</answer>\n\n"
        else:
            examples = ""
 
        for c, f in zip(claims, flags):
            text = self.prompt_template.format(examples, c, f if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_stance(self, data):
        sent1 = data["s1"]
        sent2 = data["s2"]
        questions = zip(sent1, sent2)
        stances = data["stance"]
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(sent1), self.shots, replace=False)
            for i in indices:
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += "جملة ١: "+ sent1[i] + "\n" + "جملة ٢: " + sent2[i] + "\n<answer>" + stances[i] + "</answer>\n\n"
        else:
            examples = ""
 
        for (sent1, sent2), stance in zip(questions, stances):
            q_res = "جملة ١: "+ sent1 + "\n" + "جملة ٢: " + sent2
            text = self.prompt_template.format(examples, q_res, stance if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_wsd(self, data):
        exs = data["ex"]
        words = data["word"]
        defs = data["def"]
        labels = data["label"]
        qs = zip(exs, words, defs)
        texts = []

        if self.shots > 0:
            examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
            indices = np.random.choice(len(exs), self.shots, replace=False)
            for i in indices:
                q, w, d = qs[i]
                examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
                examples += "جملة: "+ q + "\n" + "كلمة: " + w + "\n" + ":تعريف" + d + "\n\n<answer>" + labels[i] + "</answer>\n\n"
        else:
            examples = ""

        for (eg, word, de), label in zip(qs, labels):
            q_res = "جملة: "+ eg + "\n" + "كلمة: " + word + "\n" + ":تعريف" + de
            text = self.prompt_template.format(examples, q_res, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}


    def construct_prompt(self, task, lang):
        if lang == "en":
            self.prompt_template = "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            self.prompt_template += "Write a response that appropriately completes the request.\n"
            # self.prompt_template += "Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.\n"
            self.prompt_template += "Dont say anything except the answer. Give the final answer between answer tags: <answer>...</answer>.\n"
            self.prompt_template += "\n"
            self.prompt_template += "{}"
            self.prompt_template += "### Instruction:\n"
            self.prompt_template += f"{self.task_instructions[task]}\n"
            self.prompt_template += "\n"
            self.prompt_template += f"### Question:\n"
            self.prompt_template += "{}"
            self.prompt_template += "\n\n"
            self.prompt_template += f"### Response:\n"
            self.prompt_template += "{}"

        elif lang == "ar":
            self.prompt_template = "يوجد أدناه تعليمات تصف مهمة، مقترنة بإدخال يوفر سياقًا إضافيًا." + "\n"
            self.prompt_template += "اكتب الرد الذي يكمل الطلب بشكل مناسب." + "\n"
            # self.prompt_template += "قبل الإجابة، فكر جيدًا في السؤال وقم بإنشاء سلسلة من الأفكار خطوة بخطوة لضمان الحصول على إجابة منطقية ودقيقة." + "\n"
            self.prompt_template += "لا تقل أي شيء باستثناء الإجابة. أعط الإجابة النهائية بين علامات الإجابة: <answer>...</answer>.\n"
            self.prompt_template += "\n"
            self.prompt_template += "{}"
            self.prompt_template += ":تعليمات" + "###" + "\n"
            self.prompt_template += f"{self.task_instructions_ar[task]}\n"
            self.prompt_template += "\n"
            self.prompt_template += ":سؤال" + "###" + "\n"
            self.prompt_template += "{}"
            self.prompt_template += "\n\n"
            self.prompt_template += ":إجابة" + "###" + "\n"
            self.prompt_template += "{}"

        else:
            if self.logger is not None:
                self.logger(lang + " not supported")
            exit()

        if self.logger is not None:
            self.logger("PROMPT:")
            self.logger(self.prompt_template)
            self.logger("\n\n")

    def get_dataset(self, task, lang="ar"):
        self.lang = lang
        self.construct_prompt(task, lang)
        task_split = task + "_" + self.split

        if os.path.exists(self.dataset_names[task_split]) and self.dataset_names[task_split].endswith(".csv"):
            dataset = load_dataset("csv", data_files=self.dataset_names[task_split])["train"]

        elif os.path.exists(self.dataset_names[task_split]) and self.dataset_names[task_split].endswith(".tsv"):
            df = pd.read_csv(self.dataset_names[task_split], delimeter="\t")
            dataset = Dataset.from_pandas(df)["train"]

        elif os.path.exists(self.dataset_names[task_split]) and self.dataset_names[task_split].endswith(".pkl"):
            with open(self.dataset_names[task_split], 'rb') as pickle_file:
                arabic_docs=pickle.load(pickle_file)

            flat_data = []
            for url, sections in arabic_docs.items():
                for section_name, section_data in sections.items():
                    flat_data.append({
                        'input_text': section_data['document'],
                        'target_text': section_data['summary'],
                    })

            df = pd.DataFrame(flat_data)
            dataset = Dataset.from_pandas(df)

        else:
            dataset_name = self.dataset_names[task_split]
            subset_name = self.subset_names[task_split]
            dataset = load_dataset(dataset_name, subset_name, split=self.dataset_splits[task_split], trust_remote_code=True)

        self.size = dataset.num_rows
        dataset = dataset.map(self.prompt_func_map[task_split], batched = True)
        
        if self.split == "train" and self.shuffle:
            dataset = dataset.shuffle(seed=42)

        if self.logger is not None:
            self.logger("\n\n")
            self.logger("DATASET SUMMARY:")
            self.logger(str(dataset))
            self.logger("\n\n")

            self.logger("EXAMPLE DATA INSTANCE:")
            self.logger(dataset["text"][-1])
            self.logger("\n\n")
        else:
            print("\n\n")
            print("DATASET SUMMARY:")
            print(str(dataset))
            print("\n\n")

            print("EXAMPLE DATA INSTANCE:")
            print(dataset["text"][100])
            print("\n\n") 

        return dataset


if __name__ == "__main__":
    # FT_Dataset("", split="train").get_dataset("paraphrasing", "en")
    # FT_Dataset("", split="train").get_dataset("transliteration", "en")
    # FT_Dataset("", split="train").get_dataset("translation", "en")

    FT_Dataset("", split="train", shots=3).get_dataset("sentiment", "en")
