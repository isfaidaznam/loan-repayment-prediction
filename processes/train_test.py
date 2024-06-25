import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from config.config import Config
from processes import process_confusion_matrix
from processes.data_visualisation import export_accuracy_curve, export_loss_curve
from processes.formating import print_title


def get_train_test_data():
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")
    return train_df, test_df


class Model_stats:
    def __init__(self, config_file='ann_model/model_performance.yaml'):
        with open(config_file, 'r') as f:
            self.__dict__ = yaml.safe_load(f)


def export_model_if_better(model, confusion_matrix_df, history):
    intent = "a"
    try:
        newly_trained_accuracy = process_confusion_matrix.get_accuracy(confusion_matrix_df)
        newly_trained_sensitivity = process_confusion_matrix.get_sensitivity(confusion_matrix_df)
        prev_accuracy = Model_stats().ANN_PERFORMANCE["ACCURACY"]
        prev_sensitivity = Model_stats().ANN_PERFORMANCE["SENSITIVITY"]

        print_title("Performance Comparison/Improvements")
        print(f"Accuracy\t: {prev_accuracy:2%} --> {newly_trained_accuracy:2%}")
        print(f"Sensitivity\t: {prev_sensitivity:2%} --> {newly_trained_sensitivity:2%}")

        score = []
        if newly_trained_sensitivity > prev_sensitivity:
            score.append("Sensitivity")
        if newly_trained_accuracy > prev_accuracy:
            score.append("Accuracy")

        if len(score) >= 1:
            print(f"The newly trained model has better {" and ".join(score)} score...")

        while intent.lower() not in "yn":
            intent = input("Overwrite model ? [y/n]\t:")[0]
    except:
        print("Unable to read performance value from 'trained_model/ann_model/model_performance.yaml'.")
        while intent.lower() not in "yn":
            intent = input("Overwrite model ? [y/n]\t:")[0]

    if intent.lower() == "y":
        print("\nSaving newly trained model...")
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
        with open('trained_model/ann_model/model_performance.yaml', 'w') as f:
            yaml.dump(data, f, indent=4)
        print("Model Performance stored in 'trained_model/ann_model/model_performance.yaml'")

        model.save('trained_model/ann_model/predict_loan_repay_fail_model.keras')
        print("Trained Model Performance stored in 'trained_model/ann_model/predict_loan_repay_fail_model.keras'")

        import shutil
        shutil.copyfile("config/config.yaml", "trained_model/ann_model/config.yaml")

        export_accuracy_curve(history)
        export_loss_curve(history)





def train_neural_network(train_df, test_df):
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
    X_test = test_df.loc[:, column_order]
    Y_test = test_df['repay_fail']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=Config().ML_TRAINING['RANDOMNESS'],
                                                      random_state=42)

    print_title("Neural Network Training and Testing")
    model = Sequential()
    model.add(Dense(total_input_node, activation='relu', input_shape=(X.shape[1],)))
    for l in layer_config:
        act_funct = list(l.keys())[0].lower()
        num_of_node = l[list(l.keys())[0]]
        model.add(Dense(num_of_node, activation=act_funct))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Model Training
    history = model.fit(X_train,
                        y_train,
                        epochs=Config().ML_TRAINING["EPOCH"],
                        shuffle=Config().ML_TRAINING['RANDOMNESS'],
                        batch_size=32,
                        validation_data=(X_val, y_val))

    y_pred = model.predict(X_test)

    result_df = pd.DataFrame()
    result_df["target"] = Y_test
    result_df["target_binary"] = (Y_test > 0).astype(int)
    result_df["raw_predict"] = y_pred
    result_df["predict_binary"] = (y_pred > 0.5).astype(int)

    conf_matrix = confusion_matrix(result_df["target_binary"],
                                   result_df["predict_binary"])
    conf_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

    print_title("Training and Testing Result")
    print("\nConfusing Matrix\t:\n", conf_df)

    accuracy = accuracy_score(result_df["target_binary"], result_df["predict_binary"])
    sensitivity = process_confusion_matrix.get_sensitivity(conf_df)
    print("\nAccuracy\t:", accuracy)
    print("Sensitivity\t:", sensitivity)

    export_model_if_better(model, conf_df, history)