import pandas as pd
import tensorflow as tf
import yaml
from processes.formating import print_title
from processes.preprocess_data import preprocess_data, preprocess_data_predict


def read_input():
    with open('input_for_prediction.yaml', 'r') as f:
        data = yaml.safe_load(f)
    df = pd.DataFrame([data])
    return df


def get_trained_model():
    try:
        model = tf.keras.models.load_model("ann_model/predict_loan_repay_fail_model.keras")
        print_title("Model Summary")
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__=="__main__":
    try:
        print("Reading new data from 'input_for_prediction.yaml'...")
        new_data = read_input()
        preprocessed_data = preprocess_data_predict(new_data)
        model = get_trained_model()
        result = model.predict(preprocessed_data)
        # f"{row['failure_rate']:.2f}%
        repay_fail = (result[0][0] > 0.5).astype(int)
        confidence_level = (abs(result[0][0] - 0.5)) * 2 * 100
        text_result = "Fail" if repay_fail else "Not Fail"
        print_title("Prediction Result")
        print(f"Pridction class\t: {text_result}")
        print(f"Confidence Level\t: {confidence_level:.2f}%")

    except Exception as e:
        print("Unexpected error occurred during predicting")