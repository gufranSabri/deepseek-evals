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

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(inputs), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + inputs[i] + "\n\n" + self.a_head + "<answer>" + str(outputs[i]) + "</answer>\n\n"

        examples = ""
        shot_inputs = [
            "@USER @USER ديموكراتية دى ولا موش ديموكراتية يا متعلمين يا بتوع المدارس 😂😂😂",
            "@USER بس عيب اللي يصير من الجحد والنكران يا نشجعه يا نسكت اما النكران ذا عيب",
            "يا جامعية يا فاهمة و يا عاقلة ما بتموتين اذا شليتي وصاختج و قطيتيهم فالزبالة🙂",
            "RT @USER: ناموا نامت عليكوا حيطه يا مكتئبين يا أعداء الفرحه ناموا",
            "@USER يا نكدي يا طاقيه خلنا مبسوطين",
        ]
        shot_outputs = ["0", "0", "1", "1", "1"]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shot_inputs[i] + "\n\n" + self.a_head + "<answer>" + shot_outputs[i] + "</answer>\n\n"
            
        
        for text, label in zip(inputs, outputs):
            text = self.prompt_template.format(examples, text, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}
    
    def format_prompt_hate(self, data):
        inputs = data["tweet"]
        outputs = data["hate"]
        texts = []
        
        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(inputs), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + inputs[i] + "\n\n" + self.a_head + "<answer>" + str(outputs[i]) + "</answer>\n\n"

        examples = ""
        shots_inputs = [
            "يالليله يا كئيبه يا حزينه يا ثقيله عديييييي .",
            "خلوني اشوفكم يا عيال الكلب يا وضيعين يا فقر فايزين بكأس الملك وربي ما يفصل بيننا الا رب العباد",
            "@USER وانت مين يا حمساوي يا ابو أصل واطي عشان تأيد ولا ترفض",
            "@USER كل سنه وانت طيب يا خطير يا اخطر مهاجم في مصر ارمي كل حاجة من دماغك وحتشوف توفيق ربنا ليك",
            "@USER الكردي ما هذه العنصرية البغيضة يا مزورين يا كارهي دين الله يا عبيد اتاتورك"
        ]
        shots_outputs = ["0", "1", "1", "0", "1"]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shots_inputs[i] + "\n\n" + self.a_head + "<answer>" + shots_outputs[i] + "</answer>\n\n"

        for text, label in zip(inputs, outputs):
            text = self.prompt_template.format(examples, text, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_sentiment(self, data):
        inputs = data["text"]
        outputs = data["label"]
        texts = []
        
        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(inputs), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + inputs[i] + "\n\n" + self.a_head + "<answer>" + str(outputs[i]) + "</answer>\n\n"

        examples = ""
        shot_inputs = [
            "احيانا يكون الفشل دافع للنجاح",
            "اذا شعرت بشيء من حالات الاحباط والملل فجرّب العمل التطوعي",
            "اذا كنّا بالفعل على موعد مع رفع لاسعار الخبز كما يشاع، فالاولى بالحكومه اولا ان تقوم بترشيد الدعم بحيث يذهب فقط لمن يستحقه!",
            "ارحمونا صرنا نصف السياره نطلع بالباص حرام عليكم",
            "الاعتداءات على المعلمين والاطباء ... جرائم لا تقبل اي تبرير"
        ]
        shots_outputs = ["1", "1", "0", "0", "0"]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shot_inputs[i] + "\n\n" + self.a_head + "<answer>" + shots_outputs[i] + "</answer>\n\n"

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

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(tokenized_sents), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + tokenized_sents[i] + "\n\n" + self.a_head + "<answer>\n" + outputs[i] + "</answer>\n\n"

        examples = ""
        shot_inputs = [
            "يذكر ان صاحبي المركزين الاول والثاني و الثاني فقط يتأهلان الى سيدني .",
            "واضاف و أضاف التقرير انه أن ه يجرى حاليا التحقيق مع هؤلاء الاشخاص .",
            "لأن معظم المصانع التي سيتم س يتم إنشاؤها إنشاء ها داخل المناطق الحرة .",
            "وكان و كان المستهدف الذي أعلنته أعلنت ه الحكومة هو 3 ملايين طن .",
            "ومن و من المقرر ان تستمر هذه الايام حتى العشرين من الشهر الجاري .",
        ]
        shot_outputs = [
            "يذكر:VERB\nان:SYM\nصاحبي:AUX\nالمركزين:AUX\nالاول:SCONJ\nوالثاني:X\nو:DET\nالثاني:SCONJ\nفقط:ADV\nيتأهلان:VERB\nالى:PUNCT\nسيدني:PRON\n.:NOUN\n",
            "واضاف:X\nو:DET\nأضاف:VERB\nالتقرير:AUX\nانه:X\nأن:SYM\nه:PROPN\nيجرى:VERB\nحاليا:SCONJ\nالتحقيق:AUX\nمع:PUNCT\nهؤلاء:PART\nالاشخاص:AUX\n.:NOUN\nPUNCT",
            "لأن:DET\nمعظم:AUX\nالمصانع:AUX\nالتي:PART\nسيتم:X\nس:AUX\nيتم:VERB\nإنشاؤها:X\nإنشاء:AUX\nها:PROPN\nداخل:PUNCT\nالمناطق:AUX\nالحرة:SCONJ\n.:NOUN\n",
            "وكان:X\nو:DET\nكان:AUX\nالمستهدف:SCONJ\nالذي:PART\nأعلنته:X\nأعلنت:VERB\nه:PROPN\nالحكومة:AUX\nهو:PROPN\n3:ADP\nملايين:ADP\nطن:AUX\n.:NOUN\nPUNCT",
            "ومن:X\nو:DET\nمن:PUNCT\nالمقرر:SCONJ\nان:SYM\nتستمر:VERB\nهذه:PART\nالايام:AUX\nحتى:PUNCT\nالعشرين:ADP\nمن:PUNCT\nالشهر:AUX\nالجاري:SCONJ\n.:NOUN\nPUNCT"
        ]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shot_inputs[i] + "\n\n" + self.a_head + "<answer>\n" + shot_outputs[i] + "</answer>\n\n"

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

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(articles), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + articles[i] + "\n\n" + self.a_head + "<answer>" + summaries[i] + "</answer>\n\n"

        examples = ""
        shot_articles = [
            "أصدرت محكمة ألمانية اليوم الجمعة (25 يناير/ كانون الثاني 2013) عقوبات مشددة ضد مواطن ألماني وآخر نمساوي بتهمة الانتماء لتنظيم القاعدة. وحكمت محكمة العاصمة برلين على المتهم الألماني، البالغ من العمر 27 عاماً بالسجن لمدة تسعة أعوام، وعلى النمساوي، البالغ من العمر 23 عاماً بالسجن لمدة ستة أعوام وتسعة أشهر.   ووفقاً لبيانات الادعاء العام الألماني، فإن المتهم الألماني من الأعضاء المؤسسين لتنظيم ""مجاهدي طالبان الألمان"". وأدين المتهمان بالانتماء لتنظيم إرهابي في الخارج وتلقي تدريبات للقتال في منطقة الحدود الأفغانية-الباكستانية ضد جنود قوة المساعدة الأمنية الدولية (إيساف).   وبحسب الادعاء، فقد قام المتهم الألماني بنشر مقاطع فيديو تنطوي على تهديدات خلال معركة الانتخابات التشريعية في ألمانيا سنة 2009، مهدداً بنقل الجهاد إلى ألمانيا.   وتم القبض على المتهم الألماني في مارس/ آذار سنة 2011 في العاصمة النمساوية فيينا، ثم ألقت السلطات في برلين عقب ذلك بشهرين القبض على المتهم النمساوي، الذي تعرف على المتهم الألماني في أفغانستان.   والتزم المتهمان الصمت خلال جلسات المحاكمة التي استمرت حوالي عام. وجاء حكم المحكمة أقل بصورة طفيفة من مطالب الادعاء العام.   ي.أ/ ع.غ (د ب أ)",
            "دعا الرئيس الفرنسي إيمانويل ماكرون الملك السعودي سلمان بن عبد العزيز إلى رفع الحصار ""كاملا"" عن اليمن لإيصال المساعدات الإنسانية إلى البلد الذي يعاني من أزمة إنسانية هي الأخطر في العالم، وفق ما أكد الإليزيه الأربعاء (27 كانون الأول/ديسمبر 2017). وحسب الإليزيه فإن ماكرون قال للملك السعودي خلال اتصال هاتفي بينهما يوم الأحد الماضي إن فرنسا ترى أنه ""لا يوجد حل عسكري للنزاع في اليمن"" وأنه ""لا بد أن يعود الطرفان إلى طاولة المفاوضات"". وذكر ماكرون بأن فرنسا أدانت اطلاق الحوثيين صاروخا تم اعتراضه فوق الرياض في 19 كانون الأول/ديسمبر. وقال الإليزيه إن ماكرون والملك سلمان بحثا أيضا الوضع في سوريا مشيرا إلى أن ماكرون أيد ""العودة إلى عملية جنيف"" و""العمل على خطة سلام متوازنة وتأييد الحل الشامل الذي يحترم كل أطياف المجتمع"". وشكر ماكرون العاهل السعودي لمساهمته بمئة مليون يورو قوة مكافحة الإرهاب في منطقة الساحل، في حين اقترحت الرياض عقد اجتماع متابعة في كانون الثاني/يناير قبل قمة للمانحين في بروكسل في شباط/فبراير. يذكر أن ماكرون قام بزيارة خاطفة إلى الرياض في تشرين الثاني/نوفمبر والتقى ولي العهد الأمير محمد بن سلمان. ويعتزم زيارة إيران وإسرائيل وفلسطين ولبنان والأردن خلال العام 2018. فيما أكد التحالف العسكري الأسبوع الماضي أن ميناء الحديدة سيبقى مفتوحا لثلاثين يوما أمام شحنات المساعدات والبضائع التي تنقل الأغذية والوقود. ز.أ.ب/أ.ح (أ ف ب، رويترز)",
            "يلتقي المستشار النمساوي زباستيان كورتس رئيس حكومة ولاية بافاريا الألمانية ماركوس زودر اليوم (الأربعاء 20 يونيو/ حزيران 2018) في مدينة لينتس النمساوية لمناقشة سياسة التعامل مع اللاجئين على الحدود المشتركة بين النمسا وبافاريا. ويأتي هذا اللقاء، المخطط له منذ شهور، في خضم جدل حاد داخل الحكومة الألمانية حول رد لاجئين من على الحدود. ويطالب كورتس وزودر كما وزير الداخلية الألماني (هورست زيهوفر المنتمي أيضا للحزب الاجتماعي المسيحي البافاري) باتباع نهج متشدد في سياسة اللجوء. ويذكر أن ماركوس زودر انتقد خطط المستشارة ميركل والرئيس الفرنسي ماكرون بشأن وضع موازنة خاصة بمنطقة اليورو. وقال زودر، المنتمي للحزب المسيحي الاجتماعي البافاري، اليوم قبيل لقائه بالمستشار النمساوي في مدينة لينتس الألمانية ""لا يمكننا الآن طرح موزانات موازية إضافية، أو محاولة تخفيف استقرار العملة"". وحذر زودر ميركل من الخلط بين سياسة اللجوء والسياسة المالية الأوروبية، مضيفا أنه لا يجوز أن تحاول المستشارة تحفيز دول أوروبية أخرى على التعاون في قضايا اللجوء عبر تعهدات مالية، وقال ""الأمران في مجالين مختلفين. هناك حاجة لمبدأ واضح لدولة القانون"". وذكر زودر أن حزبه يطالب باستدعاء لجنة الائتلاف الحاكم لمناقشة هذا الأمر. واتفقت ميركل وماكرون أمس الثلاثاء خلال لقائهما في مدينة ميسبرغ الألمانية على وضع موازنة لمنطقة اليورو في إطار هياكل الموازنة الحالية، لكن دون إعطاء بيانات عن مقدار هذه الميزانية للعام 2021 وتهدف ميركل وماكرون من ذلك إلى جعل اليورو أكثر مقاومة للأزمات، وضخ استثمارات بالمليارات في المنطقة. تجدر الإشارة إلى أن الحزب البافاري أمهل ميركل حتى نهاية هذا الشهر للتوصل إلى اتفاق أوروبي حول سياسة اللجوء. وفي حال عدم تمكن ميركل من تحقيق ذلك، يعتزم وزير الداخلية هورست زيهوفر عدم السماح للاجئين المسجلين في دول أخرى بالاتحاد الأوروبي بعبور الحدود الألمانية. ح.ز/ م.س (د.ب.أ)",
            "مثل ثلاثة أشخاص يشتبه بأنهم من المتشددين الإسلامويين اليوم الخميس (التاسع من تشرين الثاني/نوفمبر 2017) أمام محكمة في مدينة ميونيخ الألمانية على خلفية تهم تتعلق بدعمهم ""لمنظمة إرهابية أجنبية"" في سوريا. وقال فلوريان جليفتسكي المتحدث باسم المحكمة العليا في ميونيخ إن المتهمين يعتقد بأنهم ""أمدوا (جماعة) جند الشام بسيارة إسعاف ومركبات أخرى عام 2013"". وتبلغ أعمارهم بين الـ30 و38 عاماً. وينحدر اثنان منهم من البوسنة والهرسك ويحمل الثالث الجنسية الكوسوفية. وقالت مصادر قضائية ألمانية إن جند الشام جماعة من أصل شيشاني تسعى لتأسيس خلافة إسلامية في المنطقة. ويصفها المعهد الألماني للشؤون الدولية والأمنية بأنها جماعة حاصلة على تدريب جيد دأبت على التعاون مع جبهة النصرة الفرع السابق لتنظيم القاعدة في سوريا. وفي سياق منفصل، وجهت المحكمة العليا في شتوتغارت تهمة ارتكاب جرائم حرب لجندي سابق في الجيش العراقي. وحسب المحكمة فقد عُثر في هاتف الجندي على صور تظهره وهو حاملاً رأس مقطوع لإرهابي. ويبلغ العراقي من العمر 24 عاماً. وحسب المحكمة فقد هدد العراقي لاجئ أفغاني بالقتل؛ إذ عرض عليه الصور في هاتفه قائلاً: ""سأفعل بك كما فعلت بإرهابي داعش"". خ.س/ح.ع.ح (رويترز، د ب أ)",
            "توجه رئيس الوزراء الإسرائيلي بنيامين نتنياهو اليوم الأحد إلى الولايات المتحدة في زيارة يستقبله خلالها الرئيس الأمريكي باراك أوباما في البيت الأبيض غدا (التاسع من نوفمبر/تشرين الثاني 2015). وسيكون هذا هو اللقاء الأول بينهما منذ توقيع الاتفاق النووي بين إيران والدول الست الكبرى. ووفقا للإذاعة الإسرائيلية، فقد أعربت دوائر رسمية عن الأمل في أن تفتح هذه الزيارة صفحة جديدة في العلاقات بين الجانبين. ويتوقع أن تتمحور المحادثات بين نتنياهو وأوباما حول ""حزمة المساعدات التي ستقدمها الولايات المتحدة لإسرائيل""، وكذلك ""انعدام المسار السياسي مع الطرف الفلسطيني"". وذكرت الإذاعة أن نتنياهو يسعى إلى طرح مسألة مراقبة المنشآت النووية الإيرانية إضافة إلى طلب التوصل إلى تفاهمات بشأن تبادل المعلومات الاستخبارية مع واشنطن بخصوص الملف الإيراني. وذكرت صحيفة هاآرتس الإسرائيلية اليوم الأحد أن رئيس الوزراء سيعرض خلال الاجتماع حزمة من بوادر حسن نية تجاه الفلسطينيين في الضفة الغربية وقطاع غزة. و.ب/م.س (د.ب.أ)"
        ]
        shot_summaries = [
            "أصدرت محكمة ألمانية أحكاماً مشددة بحق ألماني ونمساوي بتهمة انتمائهما إلى تنظيم القاعدة وتلقي تدريبات قتالية في أحد معسكراتها على الحدود بين أفغانستان وباكستان. الأحكام جاءت أقل مما طالب به الادعاء الألماني.",
            "كشفت الرئاسة الفرنسية أن الرئيس ماكرون طالب السعودية برفع الحصار ""كاملا"" عن اليمن لإيصال المساعدات الإنسانية. كما شدد ماكرون على أنه ""لا يوجد حل عسكري للنزاع في اليمن"" وأنه ""لا بد أن يعود الطرفان إلى طاولة المفاوضات"".",
            "موازاة مع الجهود التي تبدلها المستشارة أنغيلا ميركل بشأن بلورة سياسة أوروبية للجوء، يلتقي المستشار النمساوي برئيس ولاية بافاريا من الحزب الاجتماعي المسيحي المطالب بتشديد قواعد اللجوء فيما يشبه تحديا للمستشارة ميركل.",
            "وجهت محكمة في ميونخ تهمة دعم منظمة إرهابية في سوريا لثلاثة أشخاص يشتبه بأنهم من المتشددين الإسلامويين. كما وجهت محكمة أخرى في شتوتغارت تهمة ارتكاب جرائم حرب لجندي سابق في الجيش العراقي.",
            "يقوم رئيس الوزراء الإسرائيلي بنيامين نتنياهو بزيارة إلى واشنطن، يجتمع بها بالرئيس الأمريكي لأول مرة منذ التوقيع على الاتفاق النووي الإيراني. ومن المنتظر أن يحمل معه نتانياهو رزمة من بوادر حسن النية اتجاه الفلسطينيين."
        ]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shot_articles[i] + "\n\n" + self.a_head + "<answer>" + shot_summaries[i] + "</answer>\n\n"

        for article, summary in zip(articles, summaries):
            text = self.prompt_template.format(examples, article, summary if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_translation(self, data):
        sourceStrings = data["sourceString"]
        targetStrings = data["targetString"]
        texts = []

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(sourceStrings), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + sourceStrings[i] + "\n\n" + self.a_head + "<answer>" + targetStrings[i] + "</answer>\n\n"

        examples = ""
        shots_sourceStrings = [
            "تعليقات الحكومات على تقرير الفريق العامل المعني",
            "بمسألة إنشاء قضاء جنائي دولي",
            "٣ - قدمت استراليا في البيان الذي أدلت به في أثناء مناقشة هذا الموضوع في اللجنة السادسة في ٢٨ تشرين اﻷول/أكتوبر ١٩٩٢، تقييما للنهج العام الذي يتبعه الفريق العامل وأشارت الى أهمية العناصر التالية في ذلك النهج :",
            "ومن الجلي أن عبء العمل في المحكمة سيكون أيضا أشد محدودية، متى كانت الوﻻية التي تمارسها متفقة مع وﻻيات المحاكم الوطنية أكثر من كونها وﻻية خاصة.",
            "ويعكس هذا الموقف تفهما لعبء العمل المحدود الذي قد تواجهه المحكمة المرتآة، في سنوات عملها اﻷولى على اﻷقل، والتكاليف التي قد تتكبد نتيجة ﻹنشاء محكمة واﻹبقاء عليها كهيئة متفرغة تضم مجموعة كاملة من القضاة وهيكﻻ إداريا داعما."
        ]
        shots_targetStrings = [
            "COMMENTS OF GOVERNMENTS ON THE REPORT OF THE WORKING GROUP",
            "ON THE QUESTION OF AN INTERNATIONAL CRIMINAL JURISDICTION",
            "3. In its intervention during the debate on this issue in the Sixth Committee on 28 October 1992, Australia assessed the general approach of the Working Group and noted the importance of the following elements of that approach:",
            "he workload of a court would also clearly be more limited if it exercised concurrent jurisdiction with national courts rather than exclusive jurisdiction.",
            "This position reflects an understanding of the limited workload that a court would face, at least in its early years of operation, and the costs that would be incurred in establishing and maintaining a court on a full-time basis with a full complement of judges and a supporting administrative structure."
        ]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shots_sourceStrings[i] + "\n\n" + self.a_head + "<answer>" + shots_targetStrings[i] + "</answer>\n\n"

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

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(sentences), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + sentences[i] + "\n\n" + self.a_head + "<answer>" + paraphrases[i] + "</answer>\n\n"

        examples = ""
        shots_sentences = [
            "إذا لم يكن لديك هدف، اجعل هدفك الأول هو العثور على هدف.",
            "احرص على الحصول على ما تحب وإلا فسوف تضطر إلى قبول ما تحصل عليه.",
            "لا تنظر إلى صغر الذنب ولكن انظر إلى عظمة من عصيت.",
            "في كثير من الأحيان عليك أن تتوقع ما هو غير متوقع.",
            "اختلاف الآراء حول إلغاء المدارس التجريبية في مصر"
        ]
        shots_paraphrases = [
            "اذا لم يكن لديك هدف فاجعل هدفك الاول ايجاد واحد .",
            "اهتم بأن تحصل على ما تحبه و الا ستكون مجبراً على ان تقبل ما تحصل عليه .",
            "لا تنظر الى صغر الخطيئة و لكن انظر الى عظم من عصيت .",
            "في أحيان كثيرة عليك ان تتوقع ما ليس متوقعاً .	",
            "تباين الآراء بشأن إلغاء المدارس التجريبية في مصر"
        ]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shots_sentences[i] + "\n\n" + self.a_head + "<answer>" + shots_paraphrases[i] + "</answer>\n\n"

        texts = []
        for sent, para in zip(sentences, paraphrases):
            text = self.prompt_template.format(examples, sent, para if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    
    def format_prompt_transliteration(self,data):
        EN = data["source"]
        AR = data["transliteration"]

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(EN), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + EN[i] + "\n\n" + self.a_head + "<answer>" + AR[i] + "</answer>\n\n"

        examples = ""
        shots_EN = [
            "Btgahzo el flat!!",
            "Enty ya benty msh btrodii 3la elbta3 da abdan",
            "2a5eraaan",
            "w stress sho3'lo",
            "enty 3amlah yom 7'ames w elnas 7trg3 mn sho3'lha w try7 w tgelk",
        ]
        shots_AR = [
            "بتجهزوا الفلت!!",
            "انتي يا بنتي مش بتردي على البتاع ده ابدا",
            "أخيران",
            "وسترس شغله",
            "انتي عملاه يوم خميس والناس حترجع من شغلها وتريح وتجي لك"
        ]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shots_EN[i] + "\n\n" + self.a_head + "<answer>" + shots_AR[i] + "</answer>\n\n"
        
        texts = []
        for en, ar in zip(EN, AR):
            text = self.prompt_template.format(examples, en, ar if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_GQA(self, data):
        question = data["question_text"]
        answer = data["answers"]
        texts = []

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(question), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + question[i] + "\n\n" + self.a_head + "<answer>" + answer[i]["text"][0] + "</answer>\n\n"

        examples = ""
        shots_question = [
            "كم عدد مرات فوز الأوروغواي ببطولة كاس العالم لكرو القدم؟",
            "من هو مكتشف المرو أو الكوارتز ؟",
            "كيف يتصل الجنين بالرحم ؟",
            "أين يقع مسجد السلطان عبد المجيد؟",
            "ما عاصمة جورجيا؟"
        ]
        shots_answer = [
            "['بطولتين', 'بطولتين']",
            "['الفرنسي (بيير كوري) وأخوه (جاك)', '(بيير كوري) وأخوه (جاك)', 'بيير كوري) وأخوه (جاك']",
            "['المَشِيمَة', 'عن طريق الحبل السري']",
            "['مدينة جبيل اللبنانية', 'مدينة جبيل اللبنانية', 'مدينة جبيل اللبنانية']",
            "['تبليسي', 'تبليسي']"
        ]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shots_question[i] + "\n\n" + self.a_head + "<answer>" + shots_answer[i] + "</answer>\n\n"

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

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(question1), self.shots, replace=False)
        #     for i in indices:
        #         # examples += self.q_head + question1[i] + "\n" + question2[i] + "\n\n" + self.a_head + "<answer>" + labels[i] + "</answer>\n\n"
        #         examples += self.q_head
        #         examples += "سؤال ١: " if self.lang == "ar" else "Question 1: "
        #         examples += question1[i] + "\n"
        #         examples += "سؤال ٢: " if self.lang == "ar" else "Question 2: "
        #         examples += question2[i] + "\n\n"
        #         examples += self.a_head + "<answer>" + str(labels[i]) + "</answer>\n\n"

        shots_question1 = [
            "من أنواع العمل ؟",
            "من ماذا تتكون حجرات القلب لدى الضفدع؟",
            "ما هي أهمية موقع جوجل؟",
            "في أي عام كانت غزوة بني النضير؟",
            "ما هو العلاج الشعبي للثآليل؟",
        ]
        shots_question2 = [
            "ما هو أنواع العمل ؟",
            "كيف تكون الدورة الدموية في قلب الضفدع؟",
            "ما هو موقع جوجل؟",
            "متى غزا النبي بني النضير؟",
            "ما هو العلاج الطبي للثالول؟"
        ]
        shots_labels = ["1", "0", "0", "1", "0"]
        examples = ""
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head
                examples += "سؤال ١: " if self.lang == "ar" else "Question 1: "
                examples += shots_question1[i] + "\n"
                examples += "سؤال ٢: " if self.lang == "ar" else "Question 2: "
                examples += shots_question2[i] + "\n\n"
                examples += self.a_head + "<answer>" + str(shots_labels[i]) + "</answer>\n\n"

        for (question1, question2), label in zip(questions, labels):
            q_res = "سؤال ١: "+ question1 + "\n" + "سؤال ٢: " + question2
            text = self.prompt_template.format(examples, q_res, label if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    def format_prompt_claim(self, data):
        claims = data["claim_s"]
        flags = data["fake_flag"]
        texts = []

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(claims), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + claims[i] + "\n\n" + self.a_head + "<answer>" + str(flags[i]) + "</answer"">\n\n"

        examples = ""
        shots_claims = [
            "الحرب جنوبي السودان تهوي بقيمة الجنيه",
            "ارتفاع في أسعار الذهب عالمياً مع انخفاض الدولار",
            "برلين: لا حرب تجارية بين أميركا وأوروبا",
            "الجيش السوري يسمح للمدنيين في الغوطة الشرقية بالبقاء",
            "روسيا مستعدة لأي حرب تجارية ضد واشنطن"
        ]
        shots_flags = ["1", "0", "0", "1", "1"]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shots_claims[i] + "\n\n" + self.a_head + "<answer>" + str(shots_flags[i]) + "</answer>\n\n"

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

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(sent1), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head
        #         examples += "جملة ١: " if self.lang == "ar" else "Sentence 1: "
        #         examples += sent1[i] + "\n"
        #         examples += "جملة ٢: " if self.lang == "ar" else "Sentence 2: "
        #         examples += sent2[i] + "\n\n"
        #         examples += self.a_head + "<answer>" + str(stances[i]) + "</answer>\n\n"

        examples = ""
        shots_sent1 = [
            "العالم يترقب موقف إيراني حول هبوط أسعار النفط",
            "فنانة بريطانية تفوز بجائزة آبل العالمية للتصوير 2020",
            "الإيزيديون يحتفلون بالأربعاء الأحمر على طريقتهم",
            "توقيع مذكرة تفاهم إماراتية كورية في مجال الطاقة",
            "الأسهم الأوروبية ضحية الحرب بين واشنطن وبكين"
        ]
        shots_sent2 = [
            "تراجع مفاجئ لأسعار النفط.. وترقب عالمي لـكلمة ترامب",
            "جائزة سوني العالمية للتصوير 2018 من نصيب فنانة بريطانية",
            "ما هو الأربعاء الأحمر الذي يحتفل به الإيزيديون ؟",
            "مصدر والوكالة الكورية للطاقة توقعان مذكرة تفاهم",
            "الحرب الصينية الأميركية تطيح الأسهم الأوروبية",
        ]
        shots_stances = ["0", "0", "1", "1", "1"]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head
                examples += "جملة ١: " if self.lang == "ar" else "Sentence 1: "
                examples += shots_sent1[i] + "\n"
                examples += "جملة ٢: " if self.lang == "ar" else "Sentence 2: "
                examples += shots_sent2[i] + "\n\n"
                examples += self.a_head + "<answer>" + str(shots_stances[i]) + "</answer>\n\n"
 
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

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(exs), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head
        #         examples += "جملة: " if self.lang == "ar" else "Sentence: "
        #         examples += exs[i] + "\n"
        #         examples += "كلمة: " if self.lang == "ar" else "Word: "
        #         examples += words[i] + "\n"
        #         examples += ":تعريف" if self.lang == "ar" else "Definition: "
        #         examples += defs[i] + "\n\n"
        #         examples += self.a_head + "<answer>" + str(labels[i]) + "</answer>\n\n"

        examples = ""
        shot_words = [
            "اندس",
            "هوس",
            "عتل",
            "خبب",
            "واطن"
        ]
        shot_defs = [
            ": اندس بين الناس: اختفى، تسلل خفية بينهم",
            ": هوس السرقة: (طب) نزعة إلى السرقة في كل الحالات",
            ": عتله إلى السجن جذبه وجره بعنف :- { } .",
            ": نوع من أنواع سير الفرس بحيث تمس أقدامها الأرض بشكل متتابع",
            ": وافقه عليه"
        ]
        shot_exs = [
            ":-اندس في الفراش خشية البرد.",
            ":-تعاني من هوس السرقة على الرغم من أنها غنية.",
            "عتل بعد ذلك زنيم",
            ":-مشى خببا.",
            ":-واطنه على التعاون معه في بناء السور."
        ]
        shot_labels = ["0", "1", "0", "1", "0"]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head
                examples += "جملة: " if self.lang == "ar" else "Sentence: "
                examples += shot_exs[i] + "\n"
                examples += "كلمة: " if self.lang == "ar" else "Word: "
                examples += shot_words[i] + "\n"
                examples += ":تعريف" if self.lang == "ar" else "Definition: "
                examples += shot_defs[i] + "\n\n"
                examples += self.a_head + "<answer>" + str(shot_labels[i]) + "</answer>\n\n"

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

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(tweets), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + tweets[i] + "\n\n" + self.a_head + "<answer>" + str(dialect[i]) + "</answer>\n\n"

        examples = ""
        shot_tweets = [
            "اول مرة من عرفت معرض الكتاب زحمة ومن جهات المعرض كلها #معرض_الكويت_الدولي_للكتاب",
            "@140041Saud الرياض حي الواحةتقاطع شارع رفحاء مع طريق ابوبكر الصديق.موقعنا عبر خرائط جوجل:https://t.co/7nNz2nyzuB",
            "الأرجنتين في أرضها و بين جمهورها خسرانه من البارغواي حتى الآن، يقولون الأرجنتين منتخب في لاعبين كبار حتى بدون ميسي راح يوصلون الثلاث نهائيات",
            "وصلت صالة 5 في هذه الساعة المتأخرة. أخذت #أوبر شاب مواطن، لطف ومهنية، سيارته نظيفة، وقدم قارورة ماء.الله يحفظه لأهله، ويبارك له في رزقه.",
            "RT @Q8Pay: @Q8Pay وهذولا نفسهم وأرخص وشحن حكومي مباشر رخيص من #امازون البريطانيواذكر اذا اشتريت بأكثر من 60£ يطلع الشحن مجانيhttps://t.c…"
        ]
        shot_dialects = ["0", "1", "0", "1", "0"]

        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shot_tweets[i] + "\n\n" + self.a_head + "<answer>" + shot_dialects[i] + "</answer>\n\n"
 
        for t, d in zip(tweets, dialect):
            text = self.prompt_template.format(examples, t, d if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}
    

    def format_prompt_sarcasm(self, data):
        tweets = data["tweet"]
        sarcasm = data["sarcasm"]
        texts = []

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(tweets), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + tweets[i] + "\n\n" + self.a_head + "<answer>" + str(sarcasm[i]) + "</answer>\n\n"

        examples = ""
        shot_tweets = [
            "حتي جوجل مش مصدق اني في بيت دمياط 💔 https://t.co/GBTmGmMRiG",
            "الأسئلة التي هربنا منها أيام المجلس العسكري لا زالت تطاردنا وستظل إلى أن نواجهها ونجيب عليها",
            "تايملاين يليق بيه فيلم 'الأرهاب والكباب' بدخلة يسرا المفاجأة 😌😃",
            "@Ahmed_ALHasani ويندوز 10 ممتاز بس جلب مشاكل واجد في لاب توب منها الاضاة الي حلتها ومنها انه الجهاز يعلق على فترات اذا تركته من دون استخدام",
            "لما اشوف هيلاري كلنتون ودونالد ترمب يتهاوشون اتذكر فام the campaign"
        ]
        shot_sarcasm = ["1", "0", "1", "0", "1"]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shot_tweets[i] + "\n\n" + self.a_head + "<answer>" + shot_sarcasm[i] + "</answer>\n\n"

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
            self.prompt_template += "### Instruction:\n"
            self.prompt_template += f"{self.task_instructions[task]}\n"
            self.prompt_template += "\n"
            self.prompt_template += "{}"
            self.prompt_template += "\n"
            self.prompt_template += "-------------------\n" if self.shots>0 else ""
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
            self.prompt_template += ":تعليمات" + "###" + "\n"
            self.prompt_template += f"{self.task_instructions_ar[task]}\n"
            self.prompt_template += "\n"
            self.prompt_template += "{}"
            self.prompt_template += "\n"
            self.prompt_template += "-------------------\n" if self.shots>0 else ""
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

        self.q_head =  "### Question:\n" if self.lang == "en" else (":سؤال" + "##" + "\n")
        self.a_head = "### Response:\n" if self.lang == "en" else (":إجابة" + "##" + "\n")
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

            # save as csv
            df = pd.DataFrame(dataset)
            df.to_csv("./train.csv", index=False)

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
            print(dataset["text"][0])
            print()
            print("\n\n") 
            
            print("Length:", len(dataset["text"]))
            print("\n")

        return dataset


if __name__ == "__main__":
    # FT_Dataset("", split="test", shots=5).get_dataset("sentiment", "ar")
    # FT_Dataset("", split="train", shots=5).get_dataset("pos_tagging", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("summarization", "en")
    # FT_Dataset("", split="test", shots=5).get_dataset("translation", "ar")
    # FT_Dataset("", split="train", shots=5).get_dataset("paraphrasing", "en")
    # FT_Dataset("", split="test", shots=5).get_dataset("transliteration", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("sqs", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("stance", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("claim", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("wsd", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("GQA", "en")
    # FT_Dataset("", split="test", shots=5).get_dataset("sarcasm", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("dialect", "ar")
    FT_Dataset("", split="test", shots=5).get_dataset("hate", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("offensive", "en")

