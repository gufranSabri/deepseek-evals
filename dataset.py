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

            "sqs_train": "./data/sqs_train.csv",
            "sqs_test": "./data/sqs_test.csv",

            "stance_train": "./data/stance_train.csv",
            "stance_test": "./data/stance_test.csv",

            "claim_train": "./data/claim_train.csv",
            "claim_test": "./data/claim_test.csv",

            "wsd_train": "./data/wsd_train.csv",
            "wsd_test": "./data/wsd_test.csv",

            # "mcq_train":"aishaalansari/CIDAR100",
            # "mcq_test":"aishaalansari/CIDAR100",

            "GQA_train": "asas-ai/tydiqa-goldp-ar",
            "GQA_test": "asas-ai/tydiqa-goldp-ar",

            # "diacratization_train":"arbml/tashkeelav2",
            # "diacratization_test":"arbml/tashkeelav2",

            "sarcasm_train": "./data/sarc_dab_train.csv",
            "sarcasm_test": "./data/sarc_dab_test.csv",

            "dialect_train": "./data/sarc_dab_train.csv",
            "dialect_test":  "./data/sarc_dab_test.csv",

            "hate_train": "./data/off_hs_train.csv",
            "hate_test": "./data/off_hs_test.csv",

            "offensive_train": "./data/off_hs_train.csv",
            "offensive_test": "./data/off_hs_test.csv",
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

            "sqs_train":"train",
            "sqs_test":"test",

            "claim_train":"train",
            "claim_test":"test",

            "stance_train":"train",
            "stance_test":"test",

            # "mcq_train":"train",
            # "mcq_test":"test",

            "GQA_train": "train",
            "GQA_test": "validation",

            # "diacratization_train":"train",
            # "diacratization_test":"test",
        }

        self.subset_names = {
            "sentiment_train": None,
            "sentiment_test": None,

            # "diacratization_train": None,
            # "diacratization_test": None,

            # "mcq_train": None,
            # "mcq_test": None,

            "pos_tagging_train": "ar_padt",
            "pos_tagging_test": "ar_padt",

            "paraphrasing_train": None,
            "paraphrasing_test": None,

            "GQA_train": None,
            "GQA_test": None,
        }

        self.prompt_func_map = {
            "sentiment_train": self.format_prompt_sentiment,
            "sentiment_test": self.format_prompt_sentiment,

            # "diacratization_train": self.format_prompt_diacratization,
            # "diacratization_test": self.format_prompt_diacratization,

            # "mcq_train": self.format_prompt_mcq,
            # "mcq_test": self.format_prompt_mcq,

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

            "sqs_train": self.format_prompt_sqs,
            "sqs_test": self.format_prompt_sqs,

            "claim_train": self.format_prompt_claim,
            "claim_test": self.format_prompt_claim,

            "stance_train": self.format_prompt_stance,
            "stance_test": self.format_prompt_stance,

            "wsd_train": self.format_prompt_wsd,
            "wsd_test": self.format_prompt_wsd,

            "sarcasm_train": self.format_prompt_sarcasm,
            "sarcasm_test": self.format_prompt_sarcasm,

            "dialect_train": self.format_prompt_dialect,
            "dialect_test": self.format_prompt_dialect,

            "hate_train": self.format_prompt_hate,
            "hate_test": self.format_prompt_hate,

            "offensive_train": self.format_prompt_offensive,
            "offensive_test": self.format_prompt_offensive,
        }

        # =============================================
        self.task_instructions = {
            "summarization": "Can you summarize the following text in one sentence? Give the answer in arabic.",
            "paraphrasing": "Paraphrase the following text while keeping the meaning intact. Give the answer in arabic.",
            "offensive": "Does this text contain offensive language? Type '1' for Offensive and '0' for Not Offensive.",
            "GQA":"What is the answer for the following question?",
            
            "grammar": "Correct the grammatical errors in this sentence",
            # "grammar": "Does this sentence have any grammatical errors? If yes, provide the correction. Otherwise, re-write the sentence",
            # "grammar": "You are a professional proofreader. Read the following sentence and correct any grammatical mistakes",
        }

        self.task_instructions_ar = {
            "sentiment": "صنف مشاعر هذه الجملة كـ 0 إذا كانت سلبية و 1 إذا كانت إيجابية. قم بالاجابة باللغة العربية ",
            "translation": "ترجم الجملة الإنجليزية التالية إلى اللغة العربية",
            "transliteration": "أنت خبير في تحويل النصوص المكتوبة بالأحرف اللاتينية وفقًا لأسلوب العربيزي. حوّل النص التالي إلى الحروف العربية. قم بالاجابة باللغة العربية ",
            "dialect": "هل كُتب هذا النص باللغة العربية الفصحى أم باللهجة العامية؟ اكتب '0' إذا كان بالفصحى و'1' إذا كان بالعامية.",
            "stance": "حدد الموقف بين الجملتين المعطيتين. اختر أحد التصنيفات التالية: (0) اختلاف، (1) اتفاق، (2) غير واضح/غير مرتبط.",
            "claim": "هل هذا الادعاء زائف؟ اكتب '1' إذا كان زائفًا و'0' إذا لم يكن كذلك.",
            "wsd": "هل يتطابق المعنى المعطى مع معنى الكلمة في هذه الجملة؟ اكتب '1' إذا كان متطابقًا و'0' إذا لم يكن كذلك.",
            "sqs": "هل تمت إعادة صياغة إحدى الجملتين لتكون مكافئة للأخرى؟ أجب بـ '1' إذا كانتا معادتي الصياغة و'0' إذا لم تكونا كذلك.",
            "hate": "صنف هذا النص كـ 0 إذا لم يكن يحتوي على خطاب كراهية و 1 إذا كان يحتوي على خطاب كراهية",
            "pos_tagging": "ما هو النوع الصرفي الصحيح لكل كلمة في هذه الجملة؟ حدد الوسم المناسب لكل كلمة من بين الخيارات التالية: ['NOUN', 'PUNCT', 'ADP', 'NUM', 'SYM', 'SCONJ', 'ADJ', 'PART', 'DET', 'CCONJ', 'PROPN', 'PRON', 'X', 'ADV', 'INTJ', 'VERB', 'AUX'].",
            "sarcasm": "صنف هذا النص كـ 0 إذا لم يكن ساخراً و 1 إذا كان ساخراً",

            "grammar": "صحح الأخطاء النحوية في هذه الجملة",
            # "grammar": "هل تحتوي هذه الجملة على أخطاء نحوية؟ إذا كانت الإجابة نعم، قم بتصحيح الجملة. ان كانت لا تحتوي على اخطاء قم باعادة كتابة الجملة.",
            # "grammar": "أنت مدقق لغوي محترف. اقرأ الجملة التالية وصحح أي أخطاء نحوية"
        }
        # =============================================


        self.size = -1

    def get_size(self):
        assert self.size > 0, "Call get_dataset() first !!!"
        return self.size
    
    def format_prompt_offensive(self, data):
        inputs = data["tweet"]
        outputs = data["offensive"]
        texts = []

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(inputs), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + inputs[i] + "\n\n" + self.a_head + "<answer>" + str(outputs[i]) + "</answer>\n\n"
        
        for text, label in zip(inputs, outputs):
            text = self.prompt_template.format(examples, text, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}
    
    def format_prompt_hate(self, data):
        inputs = data["tweet"]
        outputs = data["hate"]
        texts = []
        
        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(inputs), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + inputs[i] + "\n\n" + self.a_head + "<answer>" + str(outputs[i]) + "</answer>\n\n"

        for text, label in zip(inputs, outputs):
            text = self.prompt_template.format(examples, text, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_sentiment(self, data):
        inputs = data["text"]
        outputs = data["label"]
        texts = []
        
        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(inputs), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + inputs[i] + "\n\n" + self.a_head + "<answer>" + str(outputs[i]) + "</answer>\n\n"

        for text, label in zip(inputs, outputs):
            text = self.prompt_template.format(examples, text, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}
        

    # def format_prompt_diacratization(self, data):
    #     inputs = data["text"]
    #     outputs = data["diacratized"]
    #     texts = []

    #     examples = ""

    #     for text, diacratized in zip(inputs, outputs):
    #         text = self.prompt_template.format(examples, text, diacratized if not self.test_mode else "") + self.EOS_TOKEN
    #         texts.append(text)
        
    #     return {"text": texts}

    # def format_prompt_mcq(self, data):
    #     question = data["Question"]
    #     A, B, C, D = data["A"], data["B"], data["C"], data["D"]
    #     answers = data["answer"]
    #     texts = []

    #     examples = ""
    #     if self.shots > 0:
    #         examples = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
    #         indices = np.random.choice(len(question), self.shots, replace=False)
    #         for i in indices:
    #             examples += f"Example Question:" if self.lang == "en" else "سؤال المثال:"
    #             examples += question[i] + "\n" + A[i] + "\n" + B[i] + "\n" + C[i] + "\n" + D[i] + "\n<answer>" + answers[i] + "</answer>\n\n"

    #     for question, a, b, c, d, answer in zip(question, A, B, C, D, answers):
    #         text = self.prompt_template.format(examples, question+"\n"+a+"\n"+b+"\n"+c+"\n"+d, answer if not self.test_mode else "") + self.EOS_TOKEN
    #         texts.append(text)

    #     return {"text": texts}

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

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(tokenized_sents), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + tokenized_sents[i] + "\n\n" + self.a_head + "<answer>\n" + outputs[i] + "</answer>\n\n"

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

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(articles), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + articles[i] + "\n\n" + self.a_head + "<answer>" + summaries[i] + "</answer>\n\n"

        for article, summary in zip(articles, summaries):
            text = self.prompt_template.format(examples, article, summary if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_translation(self, data):
        sourceStrings = data["sourceString"]
        targetStrings = data["targetString"]
        texts = []

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(sourceStrings), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + sourceStrings[i] + "\n\n" + self.a_head + "<answer>" + targetStrings[i] + "</answer>\n\n"

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

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(sentences), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + sentences[i] + "\n\n" + self.a_head + "<answer>" + paraphrases[i] + "</answer>\n\n"

        texts = []
        for sent, para in zip(sentences, paraphrases):
            text = self.prompt_template.format(examples, sent, para if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    
    def format_prompt_transliteration(self,data):
        EN = data["source"]
        AR = data["transliteration"]

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(EN), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + EN[i] + "\n\n" + self.a_head + "<answer>" + AR[i] + "</answer>\n\n"
        
        texts = []
        for en, ar in zip(EN, AR):
            text = self.prompt_template.format(examples, en, ar if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_GQA(self, data):
        question = data["question_text"]
        answer = data["answers"]
        texts = []

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(question), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + question[i] + "\n\n" + self.a_head + "<answer>" + answer[i]["text"][0] + "</answer>\n\n"

        for q, a in zip(question, answer):
            text = self.prompt_template.format(examples, q, a["text"] if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_sqs(self, data):
        question1 = data["question1"]
        question2 = data["question2"]
        questions = zip(question1, question2)
        labels = data["label"]
        texts = []

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(question1), self.shots, replace=False)
            for i in indices:
                # examples += self.q_head + question1[i] + "\n" + question2[i] + "\n\n" + self.a_head + "<answer>" + labels[i] + "</answer>\n\n"
                examples += self.q_head
                examples += "سؤال ١: " if self.lang == "ar" else "Question 1: "
                examples += question1[i] + "\n"
                examples += "سؤال ٢: " if self.lang == "ar" else "Question 2: "
                examples += question2[i] + "\n\n"
                examples += self.a_head + "<answer>" + str(labels[i]) + "</answer>\n\n"

        for (question1, question2), label in zip(questions, labels):
            q_res = "سؤال ١: "+ question1 + "\n" + "سؤال ٢: " + question2
            text = self.prompt_template.format(examples, q_res, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_claim(self, data):
        claims = data["claim_s"]
        flags = data["fake_flag"]
        texts = []

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(claims), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + claims[i] + "\n\n" + self.a_head + "<answer>" + str(flags[i]) + "</answer"">\n\n"
 
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

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(sent1), self.shots, replace=False)
            for i in indices:
                examples += self.q_head
                examples += "جملة ١: " if self.lang == "ar" else "Sentence 1: "
                examples += sent1[i] + "\n"
                examples += "جملة ٢: " if self.lang == "ar" else "Sentence 2: "
                examples += sent2[i] + "\n\n"
                examples += self.a_head + "<answer>" + str(stances[i]) + "</answer>\n\n"
 
        for (sent1, sent2), stance in zip(questions, stances):
            q_res = ""
            if "en" in self.lang:
                q_res = "Sentence 1: "+ sent1 + "\n" + "Sentence 2: " + sent2
            else:
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

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(exs), self.shots, replace=False)
            for i in indices:
                examples += self.q_head
                examples += "جملة: " if self.lang == "ar" else "Sentence: "
                examples += exs[i] + "\n"
                examples += "كلمة: " if self.lang == "ar" else "Word: "
                examples += words[i] + "\n"
                examples += ":تعريف" if self.lang == "ar" else "Definition: "
                examples += defs[i] + "\n\n"
                examples += self.a_head + "<answer>" + str(labels[i]) + "</answer>\n\n"

        for (eg, word, de), label in zip(qs, labels):
            q_res = ""
            if self.lang == "en":
                q_res = "Sentence: "+ eg + "\n" + "Word: " + word + "\n" + "Definition: " + de
            else:
                q_res = "جملة: "+ eg + "\n" + "كلمة: " + word + "\n" + ":تعريف" + de
            text = self.prompt_template.format(examples, q_res, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_dialect(self, data):
        tweets = data["tweet"]
        dialect = data["dialect"]
        texts = []

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(tweets), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + tweets[i] + "\n\n" + self.a_head + "<answer>" + str(dialect[i]) + "</answer>\n\n"
 
        for t, d in zip(tweets, dialect):
            text = self.prompt_template.format(examples, t, d if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}
    

    def format_prompt_sarcasm(self, data):
        tweets = data["tweet"]
        sarcasm = data["sarcasm"]
        texts = []

        examples = ""
        if self.shots > 0:
            examples = self.e_head
            indices = np.random.choice(len(tweets), self.shots, replace=False)
            for i in indices:
                examples += self.q_head + tweets[i] + "\n\n" + self.a_head + "<answer>" + str(sarcasm[i]) + "</answer>\n\n"
 
        for t, s in zip(tweets, sarcasm):
            text = self.prompt_template.format(examples, t, s if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def construct_prompt(self, task, lang):
        if lang == "en":
            self.prompt_template = "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            self.prompt_template += "Write a response that appropriately completes the request.\n"
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
        print(self.lang, "==========================")

        self.q_head =  "### Question:\n" if self.lang == "en" else (":سؤال" + "###" + "\n")
        self.a_head = "### Answer:\n" if self.lang == "en" else (":الجواب" + "###" + "\n")
        self.e_head = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
        
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
            print(task)
            print("DATASET SUMMARY")
            print(str(dataset))
            print("\n\n")

            print("EXAMPLE DATA INSTANCE:")
            print(dataset["text"][100])
            print("\n\n") 
            
            print("Length:", len(dataset["text"]))
            print("\n")

        return dataset


# if __name__ == "__main__":
    # FT_Dataset("", split="test", shots=3).get_dataset("sentiment", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("pos_tagging", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("summarization", "en")
    # FT_Dataset("", split="test", shots=3).get_dataset("translation", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("paraphrasing", "en")
    # FT_Dataset("", split="test", shots=3).get_dataset("transliteration", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("sqs", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("stance", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("claim", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("wsd", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("GQA", "en")
    # FT_Dataset("", split="test", shots=3).get_dataset("sarcasm", "en")
    # FT_Dataset("", split="test", shots=3).get_dataset("dialect", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("hate", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("offensive", "en")

