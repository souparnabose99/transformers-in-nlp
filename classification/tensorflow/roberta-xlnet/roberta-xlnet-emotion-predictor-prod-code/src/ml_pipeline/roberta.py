
import pandas as pd
import ktrain
from ktrain import text


class RoBERTa:

    def __init__(self):
        self.model_name = "roberta-base"
        self.maxlen = 512
        self.classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.batch_size = 6

    def create_transformer(self):
        return text.Transformer(self.model_name, self.maxlen, self.classes, self.batch_size)

      
