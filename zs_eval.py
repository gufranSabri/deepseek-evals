import os
import argparse
from model import FT_Models

import sacrebleu
from rouge import Rouge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
from utils import Logger

class Eval:
    def __init__(self, task, model_name="Q1.5B", prompt_lang="ar", preds_folder="./zs_preds"):
        self.task = task
        self.model_name = model_name
        self.prompt_lang = prompt_lang
        self.preds_folder = preds_folder

        self.load_model()

        self.preds_file_path = os.path.join(self.preds_folder, "_".join([self.model_name, self.task, self.prompt_lang]))

        self.task_eval_map = {
            "sentiment": "classification",
            "pos_tagging": "classification",
            "paraphrase_detection": "classification",
            "claim": "classification",
            "stance": "classification",
            "wsd": "classification",
            "paraphrasing": "bleu",
            "transliteration": "bleu",
            "translation": "bleu",
            "summarization": "rouge",
        }

        self.eval_func_map = {
            "classification": self.classification,
            "bleu": self.bleu,
            "rouge": self.rouge
        }

        self.separator = "================================================================================="

    def evaluate(self):
        return self.eval_func_map[self.task_eval_map[self.task]]()
    

    def load_model(self):
        self.tokenizer = FT_Models(self.model_name).get_tokenizer("R1-Q1.5B")
        self.eos_token = self.tokenizer.eos_token


    def get_preds(self):
        preds_folder = "_".join([self.model_name, self.task, self.prompt_lang])
        preds_dir = os.path.join(self.preds_folder, preds_folder)

        txt_files = os.listdir(preds_dir)
        if "scores.txt" in txt_files:
            txt_files.remove("scores.txt")
        txt_files = sorted(txt_files, key=lambda x: int(x.split('.')[0]))

        self.preds = []
        self.answers = []
        # for i in range(len(txt_files)):
        #     with open(os.path.join(preds_dir, txt_files[i])) as pred_file:
        #         pred = pred_file.readlines()

        #     answer_bounds = []
        #     for i, p in enumerate(pred):
        #         if p in [self.separator, self.separator + "\n"]:
        #             answer_bounds.append(i)

        #     answer = " ".join(pred[answer_bounds[0]+1: answer_bounds[1]])
        #     self.answers.append(answer.replace("\n", ""))

        #     pred = " ".join(pred[answer_bounds[1]+1:]).replace("\n", "")
        #     answer_match = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
        #     if answer_match:
        #         self.preds.append(answer_match.group(1).strip())
        #     else:
        #         think_match = re.search(r"</think>(.*)", pred, re.DOTALL)
        #         if think_match:
        #             self.preds.append(think_match.group(1).strip())
        #         else:
        #             self.preds.append("<none>")

        for i in range(len(txt_files)):
            with open(os.path.join(preds_dir, txt_files[i])) as pred_file:
                pred = pred_file.readlines()

            answer_bounds = []
            for i, p in enumerate(pred):
                if p in [self.separator, self.separator + "\n"]:
                    answer_bounds.append(i)

            answer = " ".join(pred[answer_bounds[0]+1: answer_bounds[1]])
            self.answers.append(answer.replace("\n", ""))

            pred = " ".join(pred[answer_bounds[1]+1:]).replace("\n", "")

            # Look for </think> first to extract only what comes after it
            think_match = re.search(r"</think>(.*)", pred, re.DOTALL)
            if think_match:
                pred_after_think = think_match.group(1).strip()
            else:
                pred_after_think = pred

            # Search for answer in the extracted portion
            answer_match = re.search(r"<answer>(.*?)</answer>", pred_after_think, re.DOTALL)
            if answer_match:
                self.preds.append(answer_match.group(1).strip())
            else:
                self.preds.append("<none>")


    def classification(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        def extract_first_digit(text):
            match = re.search(r"\d", text)
            return match.group(0) if match else text 

        for i in range(len(self.preds)):
            self.preds[i] = extract_first_digit(self.preds[i].replace("\n", "").replace(" ", "").strip())
            self.answers[i] = extract_first_digit(self.answers[i].replace("\n", "").replace(self.eos_token, ""))

        # for p,a in zip(self.preds, self.answers):
        #     print(p,a)

        return self.calculate_F1(self.preds, self.answers)

    def bleu(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.eos_token, "")

        return self.calculate_bleu(self.preds, self.answers)

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
        
    def rouge(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.eos_token, "")

        return self.calculate_rouge(self.preds, self.answers)

    def calculate_F1(self, preds, answers):
        accuracy = accuracy_score(preds, answers)
        precision = precision_score(preds, answers, average='macro')
        recall = recall_score(preds, answers, average='macro')
        f1 = f1_score(preds, answers, average='macro')

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"Accuracy: {accuracy}")
        logger(f"Precision: {precision}")
        logger(f"Recall: {recall}")
        logger(f"F1 Score: {f1}")

        return accuracy, precision, recall, f1

    def calculate_bleu(self, preds, answers):
        bleu = sacrebleu.BLEU(effective_order=True)
        sentence_bleu_scores = [bleu.sentence_score(candidate, [reference]).score for reference, candidate in zip(answers, preds)]
        corpus_bleu_score = bleu.corpus_score(preds, [answers]).score
        avg_sentence_bleu_score = sum(sentence_bleu_scores) / len(sentence_bleu_scores) if sentence_bleu_scores else 0

        logger = Logger(os.path.join(self.preds_file_path, "scores.txt"))
        logger(f"Average Sentence BLEU score: {avg_sentence_bleu_score:.4f}")
        logger(f"Corpus BLEU score: {corpus_bleu_score:.4f}")

        return {
            "average_sentence_bleu": avg_sentence_bleu_score,
            "corpus_bleu": corpus_bleu_score
        }

    def calculate_rouge(self, preds, answers):
        rouge = Rouge()
        abstractive_rouge_1_scores, abstractive_rouge_2_scores, abstractive_rouge_l_scores = [], [], []
        for g_text, t_text in zip(preds, answers):
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

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',dest='model')
    parser.add_argument('--prompt_lang',dest='prompt_lang', default='ar', help='ar, en')
    parser.add_argument('--task',dest='task', default='sentiment')
    args=parser.parse_args()

    # assert args.model in ["Q1.5B", "Q7B", "Q14B"], "Invalid model!"
    assert args.prompt_lang in ["en", "ar"], "Only 'en' and 'ar' languages supported!"

    e = Eval(args.task, args.model, args.prompt_lang)
    e.evaluate()