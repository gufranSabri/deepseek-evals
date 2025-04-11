import os
import argparse

import sacrebleu
from rouge import Rouge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import FT_Models
from dataset import FT_Dataset
from utils import Logger

from evaluate import load

class Eval:
    def __init__(self, task, model_name="Q1.5B", prompt_lang="ar", preds_folder="./ft_preds"):
        self.task = task
        self.model_name = model_name
        self.prompt_lang = prompt_lang
        self.preds_folder = preds_folder

        self.read_congifs()
        self.load_tokenizer()
        self.load_data()

        self.preds_file_path = os.path.join(self.preds_folder, "_".join([self.model_name, self.task, self.prompt_lang]))

        self.task_eval_map = {
            "sentiment": "classification",
            "pos_tagging": "evaluate_postagging",
            "paraphrase_detection": "classification",
            "claim": "classification",
            "stance": "classification",
            "wsd": "classification",
            "paraphrasing": "bleu",
            "transliteration": "bleu",
            "translation": "bleu",
            "summarization": "rouge",
            "sarcasm": "classification",
            "dialect": "classification",
            "hate": "classification",
            "offensive": "classification",
            "sqs": "classification",
            "GQA": "squad"
        }

        self.eval_func_map = {
            "classification": self.classification,
            "bleu": self.bleu,
            "rouge": self.rouge,
            "squad": self.squad,
            "evaluate_postagging": self.evaluate_pos_tagging,
        }

    def evaluate(self):
        return self.eval_func_map[self.task_eval_map[self.task]]()

    def read_congifs(self):
        CONFIGS = {
            "PROMPT_LANG": "",
            "LOAD_4BIT": -1,
            "MAX_SEQ_LENGTH": -1,
        }

        file_name = "_".join([self.model_name, self.task, self.prompt_lang])+".txt"
        file_path = os.path.join("./ft_logs", file_name)

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

    def load_tokenizer(self):
        self.tokenizer = FT_Models(self.model_name).get_tokenizer(self.model_name)

    def get_preds(self):
        preds_folder = "_".join([self.model_name, self.task, self.prompt_lang])
        preds_dir = os.path.join(self.preds_folder, preds_folder)

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
    

    def squad(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        predictions = []
        references = []

        for i in range(len(self.preds)):
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "").replace("<｜end▁of▁sentence｜>", "")

            start = self.answers[i].find("[")
            end = self.answers[i].rfind("]")
            self.answers[i] = self.answers[i][start+1:end].replace('"', "")

            for j in range(len(self.preds[i])):
                self.preds[i][j] = self.preds[i][j].replace("\n", "").replace("[", "").replace("]", "")
            self.preds[i] = " ".join(self.preds[i])

            print(self.preds[i])
            print(self.answers[i])
            print()

            predictions.append({"id": str(i), "prediction_text": self.preds[i]})
            references.append({"id": str(i), "answers": {"text": [self.answers[i]], "answer_start": [0]}})

        return self.calculate_squad(predictions, references)


    def calculate_squad(self, preds, answers):
        squad_metric = load("squad")

        results = squad_metric.compute(predictions=preds, references=answers)

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"F1 Score: {str(results['f1'])}")

        return results['f1']

    def classification(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1][0]

        for i in range(len(self.answers)):
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        return self.calculate_classification(self.preds, self.answers)

    def bleu(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i][1].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

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
            self.preds[i] = self.preds[i][1].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.tokenizer.eos_token, "")

        # print(self.preds[:5])
        # print()
        # print(self.answers[:5])

        # exit()


        return self.calculate_rouge(self.preds, self.answers)

    def calculate_classification(self, preds, answers):
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
            try:
                scores = rouge.get_scores(g_text, t_text)[0]
                abstractive_rouge_1_scores.append(scores['rouge-1']['f'])
                abstractive_rouge_2_scores.append(scores['rouge-2']['f'])
                abstractive_rouge_l_scores.append(scores['rouge-l']['f'])
            except Exception as e:
                scores = rouge.get_scores(g_text[:1000], t_text[:1000])[0]
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