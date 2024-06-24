def get_tp(conf_matrix_df):
    # True Positive
    return int(conf_matrix_df["Predicted 1"]["Actual 1"])

def get_tn(conf_matrix_df):
    # False Positive
    return int(conf_matrix_df["Predicted 0"]["Actual 0"])

def get_fp(conf_matrix_df):
    # True Negative
    return int(conf_matrix_df["Predicted 1"]["Actual 0"])

def get_fn(conf_matrix_df):
    # False Negative
    return int(conf_matrix_df["Predicted 0"]["Actual 1"])

def get_sensitivity(conf_matrix_df):
    true_positive = get_tp(conf_matrix_df)
    false_negative = get_fn(conf_matrix_df)
    return float(true_positive / (true_positive + false_negative))

def get_accuracy(conf_matrix_df):
    true_positive = get_tp(conf_matrix_df)
    true_negative = get_tn(conf_matrix_df)
    total_all = true_positive + true_negative + get_fp(conf_matrix_df) + get_fn(conf_matrix_df)
    return float((true_positive + true_negative) / total_all)