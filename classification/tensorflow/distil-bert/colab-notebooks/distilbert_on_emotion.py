# -*- coding: utf-8 -*-
"""DistilBert-on-Emotion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HGjv8l4GdlQqYEqKtmE5vPRBWlPD1_mc

### Check Hardware & RAM availability:
Commands to check for available GPU and RAM allocation on runtime
"""

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

"""### References:
* https://huggingface.co/
* https://arxiv.org/abs/1907.11692

### Install Required Libraries for Transformer Models:
* Pre-Trained Transformer models are part of Hugging Face Library(transformers).
* Similarly, any datatset part of Hugging Face can be called from the datasets library.
* Finally we will use a high level abstraction package called k-train to simplify our modelling and predictions
"""

!pip install ktrain
!pip install transformers
!pip install datasets

"""### Import Libraries:"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ktrain
from ktrain import text
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datasets import list_datasets
from datasets import load_dataset
import timeit
import warnings

pd.set_option('display.max_columns', None)
warnings.simplefilter(action="ignore")

tf.__version__

"""### Load Dataset:"""

emotion_train = load_dataset('emotion', split='train')
emotion_val = load_dataset('emotion', split='validation')

print("\nTrain Dataset Features for Emotion: \n", emotion_train.features)
print("\nValidation Dataset Features for Emotion: \n", emotion_val.features)

emotion_train_df = pd.DataFrame(data=emotion_train)
emotion_val_df = pd.DataFrame(data=emotion_val)

class_label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

"""### Split Train & Validation data:"""

X_train = emotion_train_df[:]["text"]
y_train = emotion_train_df[:]["label"]
X_test = emotion_val_df[:]["text"]
y_test = emotion_val_df[:]["label"]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

"""### Instantiating a DistilBERT Instance:
Create a DistilBERT instance with the model name, max token length, the labels to be used for each category and the batch size.
"""

distilbert_transformer = text.Transformer('distilbert-base-uncased', maxlen=512, classes=class_label_names, batch_size=6)

"""### Perform Data Preprocessing:"""

distilbert_train = distilbert_transformer.preprocess_train(X_train.to_list(), y_train.to_list())
distilbert_val = distilbert_transformer.preprocess_test(X_test.to_list(), y_test.to_list())

"""### Compile AlBERT in a K-Train Learner Object:
Since we are using k-train as a high level abstration package, we need to wrap our model in a k-train Learner Object for further compuation
"""

distilbert_model = distilbert_transformer.get_classifier()

distilbert_learner_ins = ktrain.get_learner(model=distilbert_model,
                            train_data=distilbert_train,
                            val_data=distilbert_val,
                            batch_size=6)

"""### BERT Model Summary:"""

distilbert_learner_ins.model.summary()

"""### BERT Optimal Learning Rates:¶
BERT follows Knowledge Distillation on BERT, hence we can use the established batch sizes and learning rates as used in BERT:

* Batch Sizes => {16, 32}
* Learning Rates => {1e−5, 2e−5, 3e−5}
We will choose the maximum among these for our fine-tuning and evaluation purposes.

### Fine Tuning DistilBERT on Emotion Dataset:
We take our Dbpedia Ontology dataset along with the BERT model we created, define the learning-rate & epochs to be used and start fine-tuning.
"""

distilbert_fine_tuning_start= timeit.default_timer()
distilbert_learner_ins.fit_onecycle(lr=2e-5, epochs=3)
distilbert_fine_tuning_stop = timeit.default_timer()

print("\nFine-Tuning time for DistilBERT on Emotion dataset: \n", (distilbert_fine_tuning_stop - distilbert_fine_tuning_start)/60, " min")

"""### Checking DistilBERT performance metrics:"""

distilbert_validation_start= timeit.default_timer()
distilbert_learner_ins.validate()
distilbert_validation_stop= timeit.default_timer()

print("\nInference time for DistilBERT on Emotion dataset: \n", (distilbert_validation_stop - distilbert_validation_start), " sec")

distilbert_learner_ins.validate(class_names=class_label_names)

distilbert_learner_ins.view_top_losses(preproc=distilbert_transformer)

"""### Saving DistilBERT Model:"""

distilbert_predictor = ktrain.get_predictor(distilbert_learner_ins.model, preproc=distilbert_transformer)
distilbert_predictor.get_classes()

distilbert_predictor.save('/content/distilbert-predictor-on-emotion')

!zip -r /content/distilbert-predictor-on-emotion /content/distilbert-predictor-on-emotion
