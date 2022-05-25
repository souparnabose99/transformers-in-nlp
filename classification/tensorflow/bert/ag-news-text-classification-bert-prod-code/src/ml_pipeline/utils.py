
import pandas as pd
import tensorflow as tf
from datasets import list_datasets, load_dataset


def check_gpu_info():
    print("Tensorflow version : ", tf.__version__)
    print("GPU available : ", bool(tf.test.is_gpu_available))
    print("GPU name : ", tf.test.gpu_device_name())


def load_and_display_dataset_details():
    ag_news_dataset = load_dataset('ag_news')
    print("\n", ag_news_dataset)
    print("Dataset Items: \n", ag_news_dataset.items())
    print("\nDataset type: \n", type(ag_news_dataset))
    print("\nShape of dataset: \n", ag_news_dataset.shape)
    print("\nNo of rows: \n", ag_news_dataset.num_rows)
    print("\nNo of columns: \n", ag_news_dataset.num_columns)
    print("\nColumn Names: \n", ag_news_dataset.column_names)
    print("\n", ag_news_dataset.data)
    print(ag_news_dataset['train'][0])
    print(ag_news_dataset['train'][1])
    print(ag_news_dataset['train']['text'][0])
    print(ag_news_dataset['train']['label'][0])
    print()
    print(ag_news_dataset['train']['text'][35000])
    print(ag_news_dataset['train']['label'][35000])
    print()
    print(ag_news_dataset['train']['text'][60000])
    print(ag_news_dataset['train']['label'][60000])
    print()
    print(ag_news_dataset['train']['text'][100000])
    print(ag_news_dataset['train']['label'][100000])
    return None


def load_and_convert_data_to_df():
    ag_news_train = load_dataset('ag_news', split='train')
    ag_news_test = load_dataset('ag_news', split='test')
    print("Train Dataset : ", ag_news_train.shape)
    print("Test Dataset : ", ag_news_test.shape)
    print(ag_news_train[0])
    print(ag_news_test[0])
    print("\nTrain Dataset Features: \n", ag_news_train.features)
    print("\nTest Dataset Features: \n", ag_news_test.features)
    ag_news_train_df = pd.DataFrame(data=ag_news_train)
    ag_news_test_df = pd.DataFrame(data=ag_news_test)
    class_label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    print("First 10 rows of Train data : \n", ag_news_train_df.head(10))
    print("Last 10 rows of Train data : \n", ag_news_train_df.tail(10))
    print("First 10 rows of Test data : \n", ag_news_test_df.head(10))
    print("Last 10 rows of Test data : \n", ag_news_test_df.tail(10))
    print("Class Label Names: \n", class_label_names)
    return ag_news_train_df, ag_news_test_df, class_label_names

  
