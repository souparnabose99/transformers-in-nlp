
import pandas as pd
import ktrain
from ktrain import text
from datasets import list_datasets
from datasets import load_dataset
from ML_Pipeline import roberta, xlnet
from ML_Pipeline import model as Model
from ML_Pipeline import utils
from ML_Pipeline import feature_engineering

import warnings
warnings.simplefilter(action='ignore')

try:
    # Load Dataset and show details:
    print('##### Load Human Emotion Dataset and Show Details #####')
    utils.load_and_display_dataset_details()

    # Load Train, Validation & Test data and convert to DataFrame Object for further operations:
    print('##### Load Train, Validation & Test data and convert to DataFrame Object for further operations #####')
    emotion_train_df, emotion_val_df, emotion_test_df, class_label_names = utils.load_and_convert_data_to_df()

    # Run for individual models:
    models = ["roberta", "xlnet"]
    for model_name in models:
        if model_name == "roberta":
            # Data Preprocessing using K-Train:
            print('##### Data Preprocessing using K-Train for RoBERTa #####')
            roberta_transformer = roberta.RoBERTa().create_transformer()

            X_train, X_test, y_train, y_test = utils.create_train_test_split(emotion_train_df, emotion_val_df, model_name)
            roberta_train, roberta_val = feature_engineering.perform_data_preprocessing(roberta_transformer, X_train, X_test, y_train, y_test)

            # Create & Train RoBERTa Model:
            print('##### Create & Train RoBERTa Model #####')
            model_learner_ins = Model.create_and_train_model(roberta_train, roberta_val, roberta_transformer, model_name)

            # Check Model performance during training and validation:
            print('##### Check Model performance during training and validation #####')
            Model.check_model_performance(model_learner_ins, class_label_names, model_name)

            # Saving RoBERTa Model Fine-tuned on Human Emotion Dataset:
            print('##### Saving RoBERTa Model Fine-tuned on Human Emotion Dataset #####')
            Model.save_fine_tuned_model(model_learner_ins, roberta_transformer, model_name)

            # Load Fine-tuned RoBERTa Model for further predictions:
            print('##### Load Fine-tuned RoBERTa Model for further predictions #####')
            roberta_predictor = Model.load_model(model_name)

        elif model_name == "xlnet":
            # Data Preprocessing using K-Train:
            print('##### Data Preprocessing using K-Train for XLNet #####')
            xlnet_transformer = xlnet.XLNet().create_transformer()

            X_train, X_test, y_train, y_test = utils.create_train_test_split(emotion_train_df, emotion_val_df, model_name)
            xlnet_train, xlnet_val = feature_engineering.perform_data_preprocessing(xlnet_transformer, X_train, X_test, y_train, y_test)

            # Create & Train XLNet Model:
            print('##### Create & Train XLNet Model #####')
            model_learner_ins = Model.create_and_train_model(xlnet_train, xlnet_val, xlnet_transformer, model_name)

            # Check Model performance during training and validation:
            print('##### Check Model performance during training and validation #####')
            Model.check_model_performance(model_learner_ins, class_label_names, model_name)

            # Saving XLNet Model Fine-tuned on Human Emotion Dataset:
            print('##### Saving XLNet Model Fine-tuned on Human Emotion Dataset #####')
            Model.save_fine_tuned_model(model_learner_ins, xlnet_transformer, model_name)

            # Load Fine-tuned XLNet Model for further predictions:
            print('##### Load Fine-tuned XLNet Model for further predictions #####')
            xlnet_predictor = Model.load_model(model_name)

except Exception as e:
    print('!! Exception Details: !!\n', '[', e, ']')
    print('Please debug for further details')

    

