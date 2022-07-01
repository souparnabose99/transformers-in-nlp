
import pandas as pd
import ktrain
from ktrain import text
from ML_Pipeline.albert import ALBERT
from ML_Pipeline.distilbert import DistilBERT
from ML_Pipeline import model as Model
from ML_Pipeline import utils
from ML_Pipeline import feature_engineering

import warnings
warnings.simplefilter(action='ignore')

try:
    # Load Dataset and show details:
    print('##### Load Dbpedia Dataset and Show Details #####')
    dbpedia_14_train, dbpedia_14_test = utils.load_and_display_dataset_details()

    class_names_list = ['Company',
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

    # Run for individual models:
    models = ["albert", "distilbert"]
    for model_name in models:
        if model_name == "albert":
            # Data Preprocessing using K-Train:
            print('##### Data Preprocessing using K-Train for ALBERT #####')
            # albert_transformer = albert.ALBERT().create_transformer()
            albert_obj = ALBERT()
            albert_transformer = albert_obj.create_transformer()

            X_train, X_test, y_train, y_test = utils.create_train_test_split(dbpedia_14_train, dbpedia_14_test, model_name)
            albert_train, albert_val = feature_engineering.perform_data_preprocessing(albert_transformer, X_train, X_test, y_train, y_test)

            # Create & Train ALBERT Model:
            print('##### Create & Train ALBERT Model #####')
            model_learner_ins = Model.create_and_train_model(albert_train, albert_val, albert_transformer, model_name)

            # Check Model performance during training and validation:
            print('##### Check Model performance during training and validation #####')
            Model.check_model_performance(model_learner_ins, class_names_list, model_name)

            # Saving ALBERT Model Fine-tuned on Dbpedia Dataset:
            print('##### Saving ALBERT Model Fine-tuned on Dbpedia Dataset #####')
            Model.save_fine_tuned_model(model_learner_ins, albert_transformer, model_name)

        elif model_name == "distilbert":
            # Data Preprocessing using K-Train:
            print('##### Data Preprocessing using K-Train for DistilBERT #####')
            # distilbert_transformer = distilbert.DistilBERT().create_transformer()
            distilbert_obj = DistilBERT()
            distilbert_transformer = distilbert_obj.create_transformer()

            X_train, X_test, y_train, y_test = utils.create_train_test_split(dbpedia_14_train, dbpedia_14_test,
                                                                             model_name)
            distilbert_train, distilbert_val = feature_engineering.perform_data_preprocessing(distilbert_transformer, X_train,
                                                                                      X_test, y_train, y_test)

            # Create & Train DistilBERT Model:
            print('##### Create & Train DistilBERT Model #####')
            model_learner_ins = Model.create_and_train_model(distilbert_train, distilbert_val, distilbert_transformer, model_name)

            # Check Model performance during training and validation:
            print('##### Check Model performance during training and validation #####')
            Model.check_model_performance(model_learner_ins, class_names_list, model_name)

            # Saving DistilBERT Model Fine-tuned on Dbpedia Dataset:
            print('##### Saving ALBERT Model Fine-tuned on Dbpedia Dataset #####')
            Model.save_fine_tuned_model(model_learner_ins, distilbert_transformer, model_name)

except Exception as e:
    print('!! Exception Details: !!\n', '[', e, ']')
    print('Please debug for further details')

    
