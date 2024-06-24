import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from config.config import Config
from processes import process_confusion_matrix


def get_train_test_data():
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")
    return train_df, test_df


class Model_stats:
    def __init__(self, config_file='ann_model/model_performance.yaml'):
        with open(config_file, 'r') as f:
            self.__dict__ = yaml.safe_load(f)


def export_model_if_better(model, confusion_matrix_df):
    newly_trained_sensitivity = process_confusion_matrix.get_sensitivity(confusion_matrix_df)
    print(f"\n\nNewly Trained Model Sensitivity\t:{newly_trained_sensitivity}")
    try:
        prev_sensitivity = Model_stats().ANN_PERFORMANCE["SENSITIVITY"]
        print(f"Previous Model Sensitivity\t:{prev_sensitivity}")
    except:
        print("Unable to read sensitivity value from 'ann_model/model_performance.yaml'.")
        intent = "a"
        while intent.lower() not in "yn":
            intent = input("Overwrite model ? [y/n]\t:")[0]
        if intent.lower() == "y":
            prev_sensitivity = 0

    if newly_trained_sensitivity > prev_sensitivity:
        print("The newly trained model has better Sensitivity score...")
        print("Saving newly trained model...")
        data = {
            'ANN_PERFORMANCE': {
                'CONFUSION_METRIX': {
                    'TRUE_POSITIVE': process_confusion_matrix.get_tp(confusion_matrix_df),
                    'TRUE_NEGATIVE': process_confusion_matrix.get_tn(confusion_matrix_df),
                    'FALSE_POSITIVE': process_confusion_matrix.get_fp(confusion_matrix_df),
                    'FALSE_NEGATIVE': process_confusion_matrix.get_fn(confusion_matrix_df)
                },
                'ACCURACY': process_confusion_matrix.get_accuracy(confusion_matrix_df),
                'SENSITIVITY': process_confusion_matrix.get_sensitivity(confusion_matrix_df)
            }
        }
        with open('ann_model/model_performance.yaml', 'w') as f:
            yaml.dump(data, f, indent=4)
        print("Model Performance stored in 'ann_model/model_performance.yaml'")

        model.save('ann_model/predict_loan_repay_fail_model.keras')
        print("Trained Model Performance stored in 'ann_model/predict_loan_repay_fail_model.keras'")

        import shutil
        shutil.copyfile("config/config.yaml", "ann_model/config.yaml")


def train_neural_network(train_df):
    if Config().ML_TRAINING["RANDOMNESS"]:
        np.random.seed(42)
        tf.random.set_seed(42)
    # Get the column order from the config file
    column_order = Config().ML_TRAINING["INPUT"]
    layer_config = Config().ML_TRAINING["HIDDEN_LAYER"]
    total_input_node = len(column_order)

    # Reorder the columns of X
    X = train_df.loc[:, column_order]
    y = train_df['repay_fail']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=Config().ML_TRAINING['RANDOMNESS'],
                                                      random_state=42)

    # Create and train neural network ann_model
    model = Sequential()
    model.add(Dense(total_input_node, activation='relu', input_shape=(X.shape[1],)))
    for l in layer_config:
        act_funct = list(l.keys())[0].lower()
        num_of_node = l[list(l.keys())[0]]
        model.add(Dense(num_of_node, activation=act_funct))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Model Training
    model.fit(X_train,
              y_train,
              epochs=Config().ML_TRAINING["EPOCH"],
              shuffle=Config().ML_TRAINING['RANDOMNESS'],
              batch_size=32,
              validation_data=(X_val, y_val))

    y_pred = model.predict(X_val)

    result_df = pd.DataFrame()
    result_df["target"] = y_val
    result_df["target_binary"] = (y_val > 0).astype(int)
    result_df["raw_predict"] = y_pred
    result_df["predict_binary"] = (y_pred > 0.5).astype(int)

    conf_matrix = confusion_matrix(result_df["target_binary"],
                                   result_df["predict_binary"])
    conf_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    print("\nConfusing Matrix\t:\n", conf_df)

    # Calculate the accuracy and others
    accuracy = accuracy_score(result_df["target_binary"], result_df["predict_binary"])
    sensitivity = process_confusion_matrix.get_sensitivity(conf_df)
    print("\nAccuracy\t:", accuracy)
    print("Sensitivity\t:", sensitivity)

    export_model_if_better(model, conf_df)