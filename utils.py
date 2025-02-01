import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import difflib
from diacritization_evaluation import wer, der
import re
from pyarabic.araby import strip_diacritics

class Logger:
    def __init__(self, file_path):
        if not os.path.exists("/".join(file_path.split("/")[:-1])):
            os.mkdir("/".join(file_path.split("/")[:-1]))
            
        self.file_path = file_path

    def __call__(self, message):
        with open(self.file_path, "a") as f:
            f.write(f"{message}\n")
            print(message)
        

def match_diacritics(p_, e_):
    p = strip_diacritics(p_).split(" ")
    e = strip_diacritics(e_).split(" ")

    p_ = p_.split(" ")
    e_ = e_.split(" ")
    
    d = difflib.Differ()

    diff = d.compare(e, p)
    matching = list(a for a in diff)
    out = []
    i = 0
    for a in matching:
        if a.startswith("-"):
            out.append(e[i])
        elif a.startswith("+"):
            i += 1
        elif a.startswith(" "):
            out.append(p_[i])
            i += 1
    assert len(out) == len(e_) 
    return " ".join(e_), " ".join(out).strip()

def post_process(txt):
  puncts = '؟،-[]:؛;()$#@&+=_-{}"\'.\n/\\0123456789١٢٣٤٥٦٧٨٩٠١'
  out = ""
  for c in txt:
    if c in puncts:
      out += ' '
      continue
    out += c
  return re.sub(' +', ' ', out).strip()

def calculate_diacritization_score(predicted, expected):
    predicted = post_process(predicted)
    predicted, expected = match_diacritics(predicted, expected)

    der_ = der.calculate_der(
        expected,
        predicted,
        case_ending=True
    )
    wer_ = wer.calculate_wer(
        expected,
        predicted,
        case_ending=True,
    )
    der_no_ce = der.calculate_der(
        expected,
        predicted,
        case_ending=False,
    )
    wer_no_ce = wer.calculate_wer(
        expected,
        predicted,
        case_ending=False,
    )
    return der_, wer_, der_no_ce, wer_no_ce


