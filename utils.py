import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Logger:
    def __init__(self, file_path, file_mode="a"):
        if not os.path.exists("/".join(file_path.split("/")[:-1])):
            os.mkdir("/".join(file_path.split("/")[:-1]))
            
        self.file_path = file_path
        self.file_mode = file_mode

    def __call__(self, message):
        with open(self.file_path, self.file_mode) as f:
            f.write(f"{message}\n")
            print(message)