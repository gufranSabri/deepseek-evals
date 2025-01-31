
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from unsloth import FastLanguageModel

class FT_Dataset:
    def __init__(self, EOS_TOKEN, split="train", logger = None):
        self.EOS_TOKEN = EOS_TOKEN
        self.split = split
        self.logger = logger

        self.dataset_names = {
            "sentiment":"ajgt_twitter_ar",
            "diacratization":"arbml/tashkeelav2",
            "mcq":"arbml/cidar-mcq-100",
            "pos_tagging":"universal_dependencies",
            "summarization":"arbml/easc",
            "translation":"Helsinki-NLP/tatoeba_mt",
            "paraphrasing": "aishaalansari/Paraphrasing" ,
            "transliteration": "aishaalansari/Transliteration_ANETAC",
            "GQA": "asas-ai/tydiqa-goldp-ar",
        }

        self.subset_names = {
            "sentiment": None,
            "diacratization": None,
            "mcq": None,
            "pos_tagging": "ar_padt",
            "summarization": None,
            "translation": "ara-rus",
            "paraphrasing": None,
            "transliteration": None,
            "GQA": None,
        }

        self.prompt_func_map = {
            "sentiment": self.format_prompt_sentiment,
            "diacratization": self.format_prompt_diacratization,
            "mcq": self.format_prompt_mcq,
            "pos_tagging": self.format_prompt_postagging,
            "summarization": self.format_prompt_summarization,
            "translation": self.format_prompt_translation,
            "transliteration": self.format_prompt_transliteration,
            "transliteration": self.format_prompt_GQA,
        }

        self.task_instructions = {
            "sentiment": "You are an expert in sentiment analysis and natural language processing. Analyze the given text and determine whether its sentiment is positive or negative.",
            "diacratization": "You are an expert in Arabic linguistics and orthography. Given an undiacritized Arabic text, accurately restore the missing diacritics.",
            "mcq": "You are an advanced AI tutor with expertise in multiple-choice reasoning. Carefully analyze the question and provided answer choices, then select the correct answer.",
            "pos_tagging": ';You are a computational linguist specializing in syntactic analysis. Given a sentence, identify and label the part-of-speech (POS) tag for each word. Your options are ["NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", "PROPN", "PRON", "X", "ADV", "INTJ", "VERB", "AUX"]',
            "summarization": "You are a professional text summarizer with expertise in extracting key information. Read the given text and generate a concise and coherent summary that preserves the main ideas and important details.",
            "translation": "You are a multilingual translation expert proficient in Arabic and Russian. Translate the following Arabic text into fluent and grammatically correct Russian while preserving the original meaning.",
            "paraphrasing": "You are a language expert skilled in rewriting text while maintaining its original meaning. Rephrase the given sentence in a clear, natural, and grammatically correct way.",
            "transliteration": "You are a linguistic specialist skilled in phonetic transcription. Convert the given text from one script to another while preserving pronunciation as accurately as possible.",
            "GQA": "You are an advanced knowledge-based AI trained in answering general questions across multiple domains. Provide an accurate, well-structured, and informative response to the following question."
        }

        self.task_instructions_ar = {
            "sentiment": "أنت خبير في تحليل المشاعر ومعالجة اللغة الطبيعية. قم بتحليل النص المعطى وحدد ما إذا كانت مشاعره إيجابية أم سلبية.",
            "diacratization": "أنت خبير في علم اللغة والنحو العربي. إذا كان النص العربي بدون علامات تشكيل، قم باستعادة علامات التشكيل المفقودة بدقة.",
            "mcq": "أنت مدرس متقدم في مجال الذكاء الاصطناعي يتمتع بخبرة في اسئلة الاختيار من متعدد. قم بتحليل السؤال والخيارات المقدمة بعناية، ثم حدد الإجابة الصحيحة.",
            "pos_tagging": '["NOUN"، "PUNCT"، "ADP"، "NUM"، "SYM"، "SCONJ"، "ADJ"، "PART"، "DET"، "CCONJ"، "PROPN"، "PRON"، "X"، "ADV"، "INTJ"، "VERB"، "AUX"] أنت عالم لغوي حاسوبي متخصص في التحليل النحوي. إذا كان لديك جملة، حدد النوع الصرفي لكل كلمة من الكلمات. خياراتك هي',
            "summarization": "أنت متخصص في تلخيص النصوص ولديك خبرة في استخراج المعلومات الأساسية. اقرأ النص المقدم وأنشئ ملخصًا موجزًا ​​ومتماسكًا يحافظ على الأفكار الرئيسية والتفاصيل المهمة.",
            "translation": "أنت خبير في الترجمة متعددة اللغات وتتقن اللغتين العربية والروسية. قم بترجمة النص التالي من اللغة العربية إلى اللغة الروسية بشكل صحيح وسليم لغويًا، مع الحفاظ على المعنى الأصلي.",
            "paraphrasing": "أنت خبير لغوي ماهر في إعادة كتابة النص مع الحفاظ على معناه الأصلي. أعد صياغة الجملة المعطاة بطريقة واضحة وطبيعية وصحيحة نحويًا.",
            "transliteration": "أنت متخصص لغوي ماهر في النسخ الصوتي. قم بتحويل النص المعطى من نص إلى آخر مع الحفاظ على النطق بدقة قدر الإمكان.",
            "GQA": "أنت عبارة عن ذكاء اصطناعي متقدم قائم على المعرفة ومدرب على الإجابة على أسئلة عامة عبر مجالات متعددة. قدم إجابة دقيقة ومنظمة جيدًا وغنية بالمعلومات للسؤال التالي.",
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
            text = self.prompt_template.format(text, label) + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_diacratization(self, data):
        inputs = data["text"]
        outputs = data["diacratized"]
        texts = []

        for text, diacratized in zip(inputs, outputs):
            text = self.prompt_template.format(text, diacratized) + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_mcq(self, data):
        question = data["Question"]
        A, B, C, D = data["A"], data["B"], data["C"], data["D"]
        answers = data["answer"]
        texts = []

        for question, a, b, c, d, answer in zip(question, A, B, C, D, answers):
            text = self.prompt_template.format(question+"\n"+a+"\n"+b+"\n"+c+"\n"+d, answer) + self.EOS_TOKEN
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
            text = self.prompt_template.format(inp, output) + self.EOS_TOKEN
            texts.append(text)

        return {"text": texts}


    def format_prompt_summarization(self, data):
        articles = data["article"]
        summaries = data["summary"]
        texts = []

        for article, summary in zip(articles, summaries):
            text = self.prompt_template.format(article, summary) + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_translation(self, data):
        sourceStrings = data["sourceString"]
        targetStrings = data["targetString"]
        texts = []

        for sourceString, targetString in zip(sourceStrings, targetStrings):
            text = self.prompt_template.format(sourceString, targetString) + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_transliteration(self,data):
        EN = data["text"][:(data.num_rows//2)]
        AR = data["text"][data.num_rows//2:]
        texts = []
 
        for en, ar in zip(EN, AR):
            text = self.prompt_template.format(en, ar) + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_paraphrasing(self, data):
        sentences = data["First sentence"]
        paraphrases = data["second sentence"]
        texts = []
 
        for sent, para in zip(sentences, paraphrases):
            text = self.prompt_template.format(sent, para) + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_GQA(self, data):
        question = data["question_text"]
        answer = data["answers"]
        texts = []
 
        for q, a in zip(question, answer):
            text = self.prompt_template.format(q, a["text"]) + self.EOS_TOKEN
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
            self.prompt_template += "\n"
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
            self.prompt_template += "\n"
            self.prompt_template += ":إجابة" + "###" + "\n"
            self.prompt_template += "{}"

        else:
            self.logger(lang + " not supported")
            exit()

        self.logger("PROMPT:")
        self.logger(self.prompt_template)
        self.logger("\n\n")

    def get_dataset(self, task, lang="en"):
        self.construct_prompt(task, lang)

        dataset_name = self.dataset_names[task]
        subset_name = self.subset_names[task]
        dataset = load_dataset(dataset_name, subset_name, split=self.split, trust_remote_code=True)

        self.size = dataset.num_rows
        dataset = dataset.map(self.prompt_func_map[task], batched = True)

        self.logger("EXAMPLE DATA INSTANCE")
        self.logger(dataset["text"][-1])
        self.logger("\n\n")

        return dataset


if __name__ == "__main__":
    FT_Dataset("").get_dataset("sentiment")
    FT_Dataset("").get_dataset("diacratization")
    FT_Dataset("").get_dataset("mcq")
    FT_Dataset("").get_dataset("pos_tagging")
    FT_Dataset("").get_dataset("rating")
    FT_Dataset("").get_dataset("summarization")
    FT_Dataset("").get_dataset("translation")