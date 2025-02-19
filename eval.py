import warnings
import os
import shutil
import time

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import random
import os
import argparse
import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

from model import FT_Models
from dataset import FT_Dataset
from utils import Logger, calculate_diacritization_score
from unsloth import FastLanguageModel

from tqdm import tqdm
from sklearn.metrics import accuracy_score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics.bleu import BLEU
import sacrebleu

from rouge import Rouge
from diacritization_evaluation import wer, der
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class Eval:
    def __init__(self, task, model_name="Q1.5B", prompt_lang="ar", models_dir="./models", logs_dir="./logs"):
        self.task = task
        self.model_name = model_name
        self.prompt_lang = prompt_lang

        self.read_congifs()
        self.load_model()
        self.load_data()

        self.preds_file_path = os.path.join("./preds", "_".join([self.model_name, self.task, self.prompt_lang]))

    def read_congifs(self):
        CONFIGS = {
            "PROMPT_LANG": "",
            "LOAD_4BIT": -1,
            "MAX_SEQ_LENGTH": -1,
        }

        file_name = "_".join([self.model_name, self.task, self.prompt_lang])+".txt"
        file_path = os.path.join("./logs", file_name)

        with open(file_path) as log_file:
            configs = log_file.readlines()

        count = len(CONFIGS.keys())
        for c in configs:
            if c.split(":")[0] in CONFIGS.keys():
                c = c.replace("\n", "")
                try:
                    CONFIGS[c.split(":")[0]] = int(c.split(":")[1].strip())
                except:
                    CONFIGS[c.split(":")[0]] = c.split(":")[1].strip()
                count -= 1

            if count == 0:
                break

        self.CONFIGS = CONFIGS

    def load_data(self):
        self.dataset_helper = FT_Dataset(self.tokenizer.eos_token, split="test")
        self.dataset = self.dataset_helper.get_dataset(self.task, self.CONFIGS["PROMPT_LANG"])
        self.dataset_size = self.dataset_helper.get_size()

        self.answers = list(self.dataset["text"])

        for i in range(len(self.answers)):
            if self.prompt_lang == "ar":
                self.answers[i] = self.answers[i].split(":إجابة###")[1]
            else:
                self.answers[i] = self.answers[i].split("### Response:")[1]
            
            # if i == 10: break

    def load_model(self):
        self.tokenizer = FT_Models(self.model_name).get_tokenizer(self.model_name)

    def get_preds(self):
        preds_folder = "_".join([self.model_name, self.task, self.prompt_lang])
        preds_dir = os.path.join("./preds", preds_folder)

        txt_files = os.listdir(preds_dir)
        if "scores.txt" in txt_files:
            txt_files.remove("scores.txt")
        txt_files = sorted(txt_files, key=lambda x: int(x.split('.')[0]))

        preds = []
        for i in range(len(txt_files)):
            with open(os.path.join(preds_dir, txt_files[i])) as pred_file:
                pred = pred_file.readlines()

            preds.append(pred)

        self.preds = preds

    def evaluate(self):
        if self.task == "sentiment":
            return self.evaluate_sentiment()

        if self.task == "pos_tagging":
            return self.evaluate_pos_tagging()

        if self.task == "paraphrasing":
            return self.evaluate_paraphrasing()

        if self.task == "summarization":
            return self.evaluate_summarization()
        
        if self.task == "transliteration":
            return self.evaluate_transliteration()

        if self.task == "diacratization":
            return self.evaluate_diacratization()

        if self.task == "translation":
            return self.evaluate_translation()

        if self.task == "sqs":
            return self.evaluate_sqs()

        if self.task == "claim":
            return self.evaluate_claim()

        if self.task == "stance":
            return self.evaluate_stance()
        
        if self.task == "wsd":
            return self.evaluate_wsd()

    def evaluate_sentiment(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1][0]

        for i in range(len(self.answers)):
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        accuracy = accuracy_score(self.preds, self.answers)
        precision = precision_score(self.preds, self.answers, average='macro')
        recall = recall_score(self.preds, self.answers, average='macro')
        f1 = f1_score(self.preds, self.answers, average='macro')

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"Accuracy: {accuracy}")
        logger(f"Precision: {precision}")
        logger(f"Recall: {recall}")
        logger(f"F1 Score: {f1}")

        return accuracy, precision, recall, f1

    # def evaluate_sentiment(self):
    #     self.get_preds()
    #     self.answers = self.answers[:len(self.preds)]

    #     for i in range(len(self.preds)):
    #         self.preds[i] = self.preds[i][1][0]

    #     for i in range(len(self.answers)):
    #         self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

    #     accuracy = accuracy_score(self.preds, self.answers)

    #     logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
    #     logger(f"Accuracy: {accuracy}")

    #     return accuracy

    def evaluate_pos_tagging(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        total = 0
        correct = 0
        for i in range(len(self.preds)):
            pred = self.preds[i]
            answer = self.answers[i].split("\n")

            pred = pred[1:-1]
            answer = answer[1:-1]
            pred = [p.replace("\n", "") for p in pred]

            pred_tags = [token.split(":")[-1] for token in pred if ":" in token]
            true_tags = [token.split(":")[-1] for token in answer if ":" in token]

            total += len(true_tags)
            correct += sum(p == t for p, t in zip(pred_tags, true_tags))

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"Accuracy: {correct / total if total > 0 else 0.0}")

        return correct / total if total > 0 else 0.0

    def evaluate_paraphrasing(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        # Compute Sentence-Level BLEU
        bleu = sacrebleu.BLEU(effective_order=True)
        sentence_bleu_scores = [bleu.sentence_score(candidate, [reference]).score
                                for reference, candidate in zip(self.answers, self.preds)]

        # Compute Corpus-Level BLEU
        corpus_bleu_score = bleu.corpus_score(self.preds, [self.answers]).score

        # Compute Average Sentence-Level BLEU
        avg_sentence_bleu_score = sum(sentence_bleu_scores) / len(sentence_bleu_scores) if sentence_bleu_scores else 0

        # Log the scores
        logger = Logger(os.path.join(self.preds_file_path, "scores.txt"))
        logger(f"Average Sentence BLEU score: {avg_sentence_bleu_score:.4f}")
        logger(f"Corpus BLEU score: {corpus_bleu_score:.4f}")

        return {
            "average_sentence_bleu": avg_sentence_bleu_score,
            "corpus_bleu": corpus_bleu_score
        }
        
    def evaluate_summarization(self):
        rouge = Rouge()

        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        abstractive_rouge_1_scores = []
        abstractive_rouge_2_scores = []
        abstractive_rouge_l_scores = []
        for g_text, t_text in zip(self.preds, self.answers):
            scores = rouge.get_scores(g_text, t_text)[0]
            abstractive_rouge_1_scores.append(scores['rouge-1']['f'])
            abstractive_rouge_2_scores.append(scores['rouge-2']['f'])
            abstractive_rouge_l_scores.append(scores['rouge-l']['f'])

        avg_abstractive_rouge_1 = sum(abstractive_rouge_1_scores) / len(abstractive_rouge_1_scores) if abstractive_rouge_1_scores else 0
        avg_abstractive_rouge_2 = sum(abstractive_rouge_2_scores) / len(abstractive_rouge_2_scores) if abstractive_rouge_2_scores else 0
        avg_abstractive_rouge_l = sum(abstractive_rouge_l_scores) / len(abstractive_rouge_l_scores) if abstractive_rouge_l_scores else 0

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"ROUGE-1: {avg_abstractive_rouge_1}")
        logger(f"ROUGE-2: {avg_abstractive_rouge_2}")
        logger(f"ROUGE-L: {avg_abstractive_rouge_l}")

        return avg_abstractive_rouge_1, avg_abstractive_rouge_2, avg_abstractive_rouge_l


    def evaluate_transliteration(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        # Compute Sentence-Level BLEU
        bleu = sacrebleu.BLEU(effective_order=True)
        sentence_bleu_scores = [bleu.sentence_score(candidate, [reference]).score
                                for reference, candidate in zip(self.answers, self.preds)]

        # Compute Corpus-Level BLEU
        corpus_bleu_score = bleu.corpus_score(self.preds, [self.answers]).score

        # Compute Average Sentence-Level BLEU
        avg_sentence_bleu_score = sum(sentence_bleu_scores) / len(sentence_bleu_scores) if sentence_bleu_scores else 0

        # Log the scores
        logger = Logger(os.path.join(self.preds_file_path, "scores.txt"))
        logger(f"Average Sentence BLEU score: {avg_sentence_bleu_score:.4f}")
        logger(f"Corpus BLEU score: {corpus_bleu_score:.4f}")

        return {
            "average_sentence_bleu": avg_sentence_bleu_score,
            "corpus_bleu": corpus_bleu_score
        }


    # def evaluate_transliteration(self):
    #     self.get_preds()
    #     self.answers = self.answers[:len(self.preds)]

    #     for i in range(len(self.preds)):
    #         self.preds[i] = list(self.preds[i][1].replace("\n", ""))
    #         self.answers[i] = list(self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, ""))

    #     smooth_fn = SmoothingFunction().method1

    #     bleu_scores = []
    #     for reference, candidate in zip(self.answers, self.preds):
    #         bleu_score = sentence_bleu([reference], candidate, smoothing_function=smooth_fn)
    #         bleu_scores.append(bleu_score)

    #     average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    #     logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
    #     logger(f"Average BLEU score: {average_bleu_score:.4f}")

    #     return average_bleu_score

    def evaluate_translation(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        # Compute Sentence-Level BLEU
        bleu = sacrebleu.BLEU(effective_order=True)
        sentence_bleu_scores = [bleu.sentence_score(candidate, [reference]).score
                                for reference, candidate in zip(self.answers, self.preds)]

        # Compute Corpus-Level BLEU
        corpus_bleu_score = bleu.corpus_score(self.preds, [self.answers]).score

        # Compute Average Sentence-Level BLEU
        avg_sentence_bleu_score = sum(sentence_bleu_scores) / len(sentence_bleu_scores) if sentence_bleu_scores else 0

        # Log the scores
        logger = Logger(os.path.join(self.preds_file_path, "scores.txt"))
        logger(f"Average Sentence BLEU score: {avg_sentence_bleu_score:.4f}")
        logger(f"Corpus BLEU score: {corpus_bleu_score:.4f}")

        return {
            "average_sentence_bleu": avg_sentence_bleu_score,
            "corpus_bleu": corpus_bleu_score
        }

    def evaluate_sqs(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1][0]

        for i in range(len(self.answers)):
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        accuracy = accuracy_score(self.preds, self.answers)

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"Accuracy: {accuracy}")

        return accuracy

    def evaluate_claim(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1][0]

        for i in range(len(self.answers)):
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        accuracy = accuracy_score(self.preds, self.answers)

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"Accuracy: {accuracy}")

        return accuracy

    def evaluate_stance(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1][0]

        for i in range(len(self.answers)):
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        accuracy = accuracy_score(self.preds, self.answers)

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"Accuracy: {accuracy}")

        return accuracy

    def evaluate_wsd(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1][0]

        for i in range(len(self.answers)):
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        accuracy = accuracy_score(self.preds, self.answers)

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"Accuracy: {accuracy}")

        return accuracy

    def evaluate_diacratization(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        der, wer, der_no_ce, wer_no_ce, total = 0, 0, 0, 0, 0
        for original_text, predicted_text in zip(self.answers, self.preds):
            d, w, dce, wce = calculate_diacritization_score(predicted_text, original_text)
            der += d
            wer += w
            dce += dce
            wce += wce

            total += 1

        der /= total
        wer /= total
        dce /= total
        wce /= total

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"DER: {der}")
        logger(f"WER: {wer}")
        logger(f"DCE: {dce}")
        logger(f"WCE: {wce}")

        return der, wer, der_no_ce, wer_no_ce
        
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',dest='model')
    parser.add_argument('--prompt_lang',dest='prompt_lang', default='ar', help='ar, en')
    parser.add_argument('--task',dest='task', default='sentiment')
    args=parser.parse_args()

    # assert args.model in ["L8B", "L70B", "Q1.5B", "Q7B", "Q14B", "Q32B"], "Invalid model!"
    # assert args.task in ["sentiment", "diacratization", "mcq", "pos_tagging", "summarization", "translation", "paraphrasing", "transliteration", "GQA"], "Invalid Task!"
    assert args.prompt_lang in ["en", "ar"], "Only 'en' and 'ar' languages supported!"

    e = Eval(args.task, args.model, args.prompt_lang)
    e.evaluate()