
import pandas as pd
import tensorflow as tf
from datasets import list_datasets, load_dataset


def load_and_display_dataset_details():
    emotion_dataset = load_dataset('emotion')
    print("\n", emotion_dataset)
    print("Dataset Items: \n", emotion_dataset.items())
    print("\nDataset type: \n", type(emotion_dataset))
    print("\nShape of dataset: \n", emotion_dataset.shape)
    print("\nNo of rows: \n", emotion_dataset.num_rows)
    print("\nNo of columns: \n", emotion_dataset.num_columns)
    emotion_train = load_dataset('emotion', split='train')
    emotion_val = load_dataset('emotion', split='validation')
    emotion_test = load_dataset('emotion', split='test')
    print("\n\nDetails for Emotion Train Dataset: ", emotion_train.shape)
    print("Details for Emotion Validation Dataset: ", emotion_val.shape)
    print("Details for Emotion Test Dataset: ", emotion_test.shape)
    print("\nTrain Dataset Features for Emotion: \n", emotion_train.features)
    print("\nTest Dataset Features for Emotion: \n", emotion_val.features)
    print("\nTest Dataset Features for Emotion: \n", emotion_test.features)
    print(emotion_dataset['train']['text'][0])
    print(emotion_dataset['train']['label'][0])
    print()
    print(emotion_dataset['train']['text'][6000])
    print(emotion_dataset['train']['label'][6000])
    print()
    print(emotion_dataset['train']['text'][100])
    print(emotion_dataset['train']['label'][100])
    print()
    print(emotion_dataset['train']['text'][3700])
    print(emotion_dataset['train']['label'][3700])
    print()
    print(emotion_dataset['train']['text'][7100])
    print(emotion_dataset['train']['label'][7100])
    print()
    print(emotion_dataset['train']['text'][5400])
    print(emotion_dataset['train']['label'][5400])
    return None


def load_and_convert_data_to_df():
    emotion_train = load_dataset('emotion', split='train')
    emotion_val = load_dataset('emotion', split='validation')
    emotion_test = load_dataset('emotion', split='test')
    emotion_train_df = pd.DataFrame(data=emotion_train)
    emotion_val_df = pd.DataFrame(data=emotion_val)
    emotion_test_df = pd.DataFrame(data=emotion_test)
    class_label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return emotion_train_df, emotion_val_df, emotion_test_df, class_label_names


def create_train_test_split(emotion_train_df, emotion_val_df, model_name):
    X_train = emotion_train_df[:]["text"]
    y_train = emotion_train_df[:]["label"]
    X_test = emotion_val_df[:]["text"]
    y_test = emotion_val_df[:]["label"]
    if model_name == "xlnet":
        X_test = emotion_val_df[:1984]["text"]
        y_test = emotion_val_df[:1984]["label"]
    print("Train Test split details for {}: \n".format(model_name), X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

  
