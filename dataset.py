import warnings
import os
import pickle
import pandas as pd
from datasets import Dataset

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import login
from datasets import load_dataset

class FT_Dataset:
    def __init__(self, EOS_TOKEN, split="train", logger = None, test_mode=False):
        login(token="token")

        self.EOS_TOKEN = "" if test_mode else EOS_TOKEN
        self.split = split
        self.logger = logger
        self.test_mode = test_mode

        self.dataset_names = {
            "sentiment_train":"ajgt_twitter_ar",
            "sentiment_test":"ajgt_twitter_ar",

            "pos_tagging_train":"universal_dependencies",
            "pos_tagging_test":"universal_dependencies",

            "summarization_train":"./data/arabic.pkl",
            "summarization_test":"arbml/easc",

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

            "sqs_train":"train",
            "sqs_test":"test",

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

            "sqs_train": self.format_prompt_sqs,
            "sqs_test": self.format_prompt_sqs,

            "claim_train": self.format_prompt_claim,
            "claim_test": self.format_prompt_claim,

            "stance_train": self.format_prompt_stance,
            "stance_test": self.format_prompt_stance,

            "wsd_train": self.format_prompt_wsd,
            "wsd_test": self.format_prompt_wsd,
        }

        self.task_instructions = {
            "sentiment": "You are an expert in sentiment analysis and natural language processing. Analyze the given text and answer 1 if the sentiment is positive and 0 if the sentiment is negative.",
            "mcq": "You are an advanced AI tutor with expertise in multiple-choice reasoning. Carefully analyze the question and provided answer choices, then select the correct answer.",
            "pos_tagging": ';You are a computational linguist specializing in syntactic analysis. Given a sentence, identify and label the part-of-speech (POS) tag for each word. Your options are ["NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", "PROPN", "PRON", "X", "ADV", "INTJ", "VERB", "AUX"]',
            "summarization": "You are a professional text summarizer with expertise in extracting key information. Read the given text and generate a concise and coherent summary that preserves the main ideas and important details.",
            "translation": "You are a multilingual translation expert proficient in Arabic and Russian. Translate the following Arabic text into fluent and grammatically correct Russian while preserving the original meaning.",
            # "paraphrasing": "You are a language expert skilled in rewriting text while maintaining its original meaning. Rephrase the given sentence in a clear, natural, and grammatically correct way.",
            "paraphrasing": "You are a language expert skilled in rewriting text while maintaining its original meaning. Rewrite the following passage using different words and sentence structures while keeping the meaning intact.",
            "transliteration": "You are a linguistic specialist skilled in phonetic transcription. Convert the given text from one script to another while preserving pronunciation as accurately as possible.",
            "GQA": "You are an advanced knowledge-based AI trained in answering general questions across multiple domains. Provide an accurate, well-structured, and informative response to the following question.",
            "sqs": "Determine whether the following two questions have the same meaning or convey similar intent. Respond with '1' if they are semantically similar and '0' otherwise. Provide a brief explanation if needed.",
            "claim": "Analyze the following claim and determine whether it is true or false. Provide a '0' or '1' response, and briefly justify your answer based on factual accuracy, logical consistency, or available evidence.",
            "stance": "Compare the stance expressed in the following two sentences. Determine whether they agree, disagree, or have an unrelated/unclear stance. Respond with '1' if both sentences support the same claim, '0' if they take opposing positions, or '2' if the stance is unclear or unrelated.",
            "wsd": "You are given an example sentence, its the word to focus on, and the word's sense/definition. Your task is to determine whether the given definition correctly represents the meaning of the word in the provided example. Predict 1 if the word sense matches the meaning in the example. Predict 0 if the word sense does not match the meaning in the example.",
        }

        self.task_instructions_ar = {
            "sentiment": "أنت خبير في تحليل المشاعر ومعالجة اللغة الطبيعية. قم بتحليل النص المعطى وأجب بـ 1 إذا كانت المشاعر إيجابية و0 إذا كانت المشاعر سلبية.",
            "mcq": "أنت مدرس متقدم في مجال الذكاء الاصطناعي يتمتع بخبرة في اسئلة الاختيار من متعدد. قم بتحليل السؤال والخيارات المقدمة بعناية، ثم حدد الإجابة الصحيحة.",
            "pos_tagging": '["NOUN"، "PUNCT"، "ADP"، "NUM"، "SYM"، "SCONJ"، "ADJ"، "PART"، "DET"، "CCONJ"، "PROPN"، "PRON"، "X"، "ADV"، "INTJ"، "VERB"، "AUX"] أنت عالم لغوي حاسوبي متخصص في التحليل النحوي. إذا كان لديك جملة، حدد النوع الصرفي لكل كلمة من الكلمات. خياراتك هي',
            "summarization": "أنت متخصص في تلخيص النصوص ولديك خبرة في استخراج المعلومات الأساسية. اقرأ النص المقدم وأنشئ ملخصًا موجزًا ​​ومتماسكًا يحافظ على الأفكار الرئيسية والتفاصيل المهمة.",
            "translation": "أنت خبير في الترجمة متعددة اللغات وتتقن اللغتين العربية والروسية. قم بترجمة النص التالي من اللغة العربية إلى اللغة الروسية بشكل صحيح وسليم لغويًا، مع الحفاظ على المعنى الأصلي.",
            "paraphrasing": "أنت خبير لغوي ماهر في إعادة كتابة النص مع الحفاظ على معناه الأصلي. أعد صياغة الجملة المعطاة بطريقة واضحة وطبيعية وصحيحة نحويًا.",
            "transliteration": "أنت متخصص لغوي ماهر في النسخ الصوتي. قم بتحويل النص المعطى من نص إلى آخر مع الحفاظ على النطق بدقة قدر الإمكان.",
            "GQA": "أنت عبارة عن ذكاء اصطناعي متقدم قائم على المعرفة ومدرب على الإجابة على أسئلة عامة عبر مجالات متعددة. قدم إجابة دقيقة ومنظمة جيدًا وغنية بالمعلومات للسؤال التالي.",
            "sqs": 'حدد ما إذا كان السؤالان التاليان لهما نفس المعنى أو ينقلان نفس القصد. أجب بـ "1" إذا كانا متشابهين دلاليًا و"0" بخلاف ذلك. قدم شرحًا موجزًا ​​إذا لزم الأمر.',
            "claim": 'قم بتحليل الادعاء التالي وحدد ما إذا كان صحيحًا أم خاطئًا. أعطِ إجابة "0" أو "1"، وبرر إجابتك بإيجاز استنادًا إلى دقة الحقائق أو الاتساق المنطقي أو الأدلة المتاحة.',
            "stance": 'قارن بين الموقفين المعبر عنهما في الجملتين التاليتين. حدد ما إذا كانا متفقين أو مختلفين أو لديهما موقف غير ذي صلة/غير واضح. أجب بـ "1" إذا كانت الجملتان تدعمان نفس الادعاء، أو "0" إذا كانتا تتخذان مواقف متعارضة، أو "2" إذا كان الموقف غير واضح أو غير ذي صلة.',
            "wsd": 'لقد تم إعطاؤك جملة نموذجية، وهي الكلمة التي يجب التركيز عليها، ومعنى الكلمة/تعريفها. مهمتك هي تحديد ما إذا كان التعريف المعطى يمثل بشكل صحيح معنى الكلمة في المثال المقدم. توقع 1 إذا كان معنى الكلمة يتطابق مع المعنى في المثال. توقع 0 إذا كان معنى الكلمة لا يتطابق مع المعنى في المثال.'
        }

        self.size = -1

    def get_size(self):
        assert self.size > 0, "Call get_dataset() first !!!"
        return self.size

    def format_prompt_sentiment(self, data):
        inputs = data["text"]
        outputs = data["label"]
        texts = []

        for text, label in zip(inputs, outputs):
            text = self.prompt_template.format(text, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_diacratization(self, data):
        inputs = data["text"]
        outputs = data["diacratized"]
        texts = []

        for text, diacratized in zip(inputs, outputs):
            text = self.prompt_template.format(text, diacratized if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_mcq(self, data):
        question = data["Question"]
        A, B, C, D = data["A"], data["B"], data["C"], data["D"]
        answers = data["answer"]
        texts = []

        for question, a, b, c, d, answer in zip(question, A, B, C, D, answers):
            text = self.prompt_template.format(question+"\n"+a+"\n"+b+"\n"+c+"\n"+d, answer if not self.test_mode else "") + self.EOS_TOKEN
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

        for inp, output in zip(tokenized_sents, outputs):
            text = self.prompt_template.format(inp, output if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)

        return {"text": texts}


    def format_prompt_summarization(self, data):
        X_col = "article" if self.split == "test" else "input_text"
        y_col = "summary" if self.split == "test" else "target_text"        

        articles = data[X_col]
        summaries = data[y_col]
        texts = []

        for article, summary in zip(articles, summaries):
            text = self.prompt_template.format(article, summary if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_translation(self, data):
        sourceStrings = data["sourceString"]
        targetStrings = data["targetString"]
        texts = []

        for sourceString, targetString in zip(sourceStrings, targetStrings):
            text = self.prompt_template.format(sourceString, targetString if not self.test_mode else "") + self.EOS_TOKEN
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

        texts = []
        for sent, para in zip(sentences, paraphrases):
            text = self.prompt_template.format(sent, para if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    
    def format_prompt_transliteration(self,data):
        EN = data["source"]
        AR = data["transliteration"]
        
        texts = []
        for en, ar in zip(EN, AR):
            text = self.prompt_template.format(en, ar if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_GQA(self, data):
        question = data["question_text"]
        answer = data["answers"]
        texts = []
 
        for q, a in zip(question, answer):
            text = self.prompt_template.format(q, a["text"] if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_sqs(self, data):
        question1 = data["question1"]
        question2 = data["question2"]
        questions = zip(question1, question2)
        labels = data["label"]
        texts = []
 
        for (question1, question2), label in zip(questions, labels):
            q_res = "سؤال ١: "+ question1 + "\n" + "سؤال ٢: " + question2
            text = self.prompt_template.format(q_res, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_claim(self, data):
        claims = data["claim_s"]
        flags = data["fake_flag"]
        texts = []
 
        for c, f in zip(claims, flags):
            text = self.prompt_template.format(c, f if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_stance(self, data):
        sent1 = data["s1"]
        sent2 = data["s2"]
        questions = zip(sent1, sent2)
        stances = data["stance"]
        texts = []
 
        for (sent1, sent2), stance in zip(questions, stances):
            q_res = "جملة ١: "+ sent1 + "\n" + "جملة ٢: " + sent2
            text = self.prompt_template.format(q_res, stance if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}


    def format_prompt_wsd(self, data):
        exs = data["ex"]
        words = data["word"]
        defs = data["def"]
        labels = data["label"]
        qs = zip(exs, words, defs)
        texts = []
 
        for (eg, word, de), label in zip(qs, labels):
            q_res = "جملة: "+ eg + "\n" + "كلمة: " + word + "\n" + ":تعريف" + de
            text = self.prompt_template.format(q_res, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}


    def construct_prompt(self, task, lang):
        if lang == "en":
            self.prompt_template = "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            self.prompt_template += "Write a response that appropriately completes the request.\n"
            self.prompt_template += "Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.\n"
            self.prompt_template += "\n"
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
            self.prompt_template = "قبل الإجابة، فكر جيدًا في السؤال وقم بإنشاء سلسلة من الأفكار خطوة بخطوة لضمان الحصول على إجابة منطقية ودقيقة." + "\n"
            self.prompt_template += "\n"
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
        
        if self.split == "train":
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
            print(dataset["text"][-1])
            print("\n\n") 

        return dataset


if __name__ == "__main__":
    FT_Dataset("", split="test").get_dataset("sqs")