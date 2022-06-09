
# -*- coding: utf-8 -*-
"""RoBERTa-Emotion-Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b8Zy_YIKIzPfDQaZ-63FufFFXRAinD5e

### Check Hardware & RAM availability:

Commands to check for available GPU and ram allocation on runtime
"""

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ktrain
from ktrain import text
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datasets import list_datasets
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import timeit
import warnings

pd.set_option('display.max_columns', None)
warnings.simplefilter(action="ignore")

"""### Loading Emotion Dataset:"""

emotion_train = load_dataset('emotion', split='train')
emotion_val = load_dataset('emotion', split='validation')
emotion_test = load_dataset('emotion', split='test')
print("Details for Emotion Train Dataset: ", emotion_train.shape)
print("Details for Emotion Validation Dataset: ", emotion_val.shape)
print("Details for Emotion Test Dataset: ", emotion_test.shape)

print("\nTrain Dataset Features for Emotion: \n", emotion_train.features)
print("\nTest Dataset Features for Emotion: \n", emotion_val.features)
print("\nTest Dataset Features for Emotion: \n", emotion_test.features)

emotion_train_df = pd.DataFrame(data=emotion_train)
emotion_val_df = pd.DataFrame(data=emotion_val)

class_label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

"""### Instantiating a RoBERTa Instance:"""

roberta_transformer = text.Transformer('roberta-base', maxlen=512, classes=class_label_names, batch_size=6)

X_train = emotion_train_df[:]["text"]
y_train = emotion_train_df[:]["label"]
X_test = emotion_val_df[:]["text"]
y_test = emotion_val_df[:]["label"]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

roberta_train = roberta_transformer.preprocess_train(X_train.to_list(), y_train.to_list())
roberta_val = roberta_transformer.preprocess_test(X_test.to_list(), y_test.to_list())

"""### Compile RoBERTa in a K-Train Learner Object:"""

roberta_model = roberta_transformer.get_classifier()

roberta_learner_ins = ktrain.get_learner(model=roberta_model,
                            train_data=roberta_train,
                            val_data=roberta_val,
                            batch_size=6)

"""### RoBERTa Model Details:"""

roberta_learner_ins.model.summary()

"""### Find Optimal Learning Rate for RoBERTa:"""

rate_finder_start_time = timeit.default_timer()
roberta_learner_ins.lr_find(show_plot=True, max_epochs=3)
rate_finder_stop_time = timeit.default_timer()

print("\nTotal time in minutes on estimating optimal learning rate: \n", (rate_finder_stop_time - rate_finder_start_time)/60)

"""### RoBERTa Optimal Learning Rates:

As per the evaluations made in the research paper "**RoBERTa: A Robustly Optimized BERT Pretraining Approach**", below are the best choices in terms of fine-tuning the model:

* Batch Sizes => {16, 32}
* Learning Rates => {1e−5, 2e−5, 3e−5}

We will choose the maximum among these for our fine-tuning and evaluation purposes.

### Fine Tuning RoBERTa on Emotion Dataset:
"""

roberta_fine_tune_start_time = timeit.default_timer()
roberta_learner_ins.fit_onecycle(lr=3e-5, epochs=3)
roberta_fine_tune_stop_time = timeit.default_timer()

print("\nTotal time in minutes for Fine-Tuning RoBERTa on Emotion Dataset: \n", (roberta_fine_tune_stop_time - roberta_fine_tune_start_time)/60)

"""### Checking RoBERTa performance metrics:"""

roberta_learner_ins.validate()

roberta_learner_ins.validate(class_names=class_label_names)

roberta_learner_ins.view_top_losses(preproc=roberta_transformer)

"""### Saving RoBERTa Model:"""

roberta_predictor = ktrain.get_predictor(roberta_learner_ins.model, preproc=roberta_transformer)
roberta_predictor.get_classes()

roberta_predictor.save('/content/roberta-emotion-predictor')

!zip -r /content/roberta-emotion-predictor /content/roberta-emotion-predictor

"""### Loading Saved Model for New Predictions:"""

roberta_predictor_new = ktrain.load_predictor('/content/roberta-emotion-predictor')
roberta_predictor_new.get_classes()

emotion_test_df = pd.DataFrame(data=emotion_test)
print("\nShape of Test Dataset: ", emotion_test_df.shape,"\n\n")
emotion_test_df.head()

emotion_test_df.info()

label_dict = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
emotion_test_df["label"] = emotion_test_df["label"].map(label_dict)
emotion_test_df.head()

emotion_test_df[emotion_test_df.columns] = emotion_test_df[emotion_test_df.columns].astype(str)

X_test_new = emotion_test_df[:]["text"]
y_test_new = emotion_test_df[:]["label"]
print(X_test_new.shape, y_test_new.shape)

test_predictions = roberta_predictor_new.predict(X_test_new.to_list())

print(confusion_matrix(y_test_new, test_predictions))

print(classification_report(y_test_new, test_predictions))

