import pandas as pd
import ktrain
from ktrain import text


class ALBERT:

    def __init__(self):
        self.model_name = "albert-base-v1"
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
        self.batch_size = 6

    def create_transformer(self):
        return text.Transformer(self.model_name, self.maxlen, self.classes, self.batch_size)

