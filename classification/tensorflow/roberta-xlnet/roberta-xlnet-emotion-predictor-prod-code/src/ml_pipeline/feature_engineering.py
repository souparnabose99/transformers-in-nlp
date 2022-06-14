
import ktrain
from ktrain import text


def perform_data_preprocessing(transformer, X_train, y_train, X_test, y_test):
    train = transformer.preprocess_train(X_train.to_list(), y_train.to_list())
    val = transformer.preprocess_test(X_test.to_list(), y_test.to_list())
    return train, val

  
