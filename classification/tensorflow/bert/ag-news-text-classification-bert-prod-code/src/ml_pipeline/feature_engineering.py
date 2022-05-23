
import ktrain
from ktrain import text


def perform_feature_engineering(ag_news_train_df, ag_news_test_df, max_length):
    (X_train, y_train), (X_test, y_test), preprocessing_var = text.texts_from_df(train_df=ag_news_train_df,
                                                                                 text_column='text',
                                                                                 label_columns='label',
                                                                                 val_df=ag_news_test_df,
                                                                                 maxlen=max_length,
                                                                                 preprocess_mode='bert')

    return (X_train, y_train), (X_test, y_test), preprocessing_var
