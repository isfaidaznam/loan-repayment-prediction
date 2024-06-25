import pandas as pd
import tensorflow as tf
import yaml
from joblib import load

from processes.formating import print_title
from processes.preprocess_data import preprocess_data, preprocess_data_predict


def read_input():
    with open('input_for_prediction.yaml', 'r') as f:
        data = yaml.safe_load(f)
    df = pd.DataFrame([data])
    return df


def get_trained_model(selection = None):
    try:
        if selection.lower() == "ann":
            model = tf.keras.models.load_model("trained_model/ann_model/predict_loan_repay_fail_model.keras")
            print_title("Model Summary")
            print(model.summary())
        elif selection.lower() == "knn":
            model = load('trained_model/knn_model/predict_loan_repay_fail_model.joblib')
            print_title("Model Summary")
            print(model.get_params())
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def select_model():
    selection = input("[a] Artificial neural Network\n[k] K-Nearest Neighbor\nSelect model for prediction [a/k]:")
    model = None
    while model not in ["ann", "knn"]:
        if selection.lower()[0] == "a":
            model = "ann"
        elif selection.lower()[0] == "k":
            model = "knn"
    return model


if __name__=="__main__":
    try:
        print("Reading new data from 'input_for_prediction.yaml'...")
        new_data = read_input()
        selection = select_model()
        preprocessed_data = preprocess_data_predict(new_data, selection)
        model = get_trained_model(selection)
        result = model.predict(preprocessed_data)
        result = result[0][0] if selection == "ann" else result[0]
        repay_fail = (result > 0.5).astype(int)
        confidence_level = (abs(result - 0.5)) * 2 * 100 if selection == "ann" else model.predict_proba(preprocessed_data)[0][repay_fail]*100
        text_result = "Fail" if repay_fail else "Not Fail"
        print_title("Prediction Result")
        print(f"Pridction class\t: {text_result}")
        print(f"Confidence Level\t: {confidence_level:.2f}%")

    except Exception as e:
        print("Unexpected error occurred during predicting")