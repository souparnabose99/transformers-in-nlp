
import pandas as pd
import ktrain
from ktrain import text


class DistilBERT:

    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.maxlen = 512
        self.classes = ['Company',
                        'EducationalInstitution',
                        'Artist',
                        'Athlete',
                        'OfficeHolder',
                        'MeanOfTransportation',
                        'Building',
                        'NaturalPlace',
                        'Village',
                        'Animal',
                        'Plant',
                        'Album',
                        'Film',
                        'WrittenWork']
        self.batch_size = 16

    def create_transformer(self):
        return text.Transformer(self.model_name, self.maxlen, self.classes, self.batch_size)

