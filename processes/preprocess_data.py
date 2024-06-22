import math
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def read_data() -> pd.DataFrame:
    try:
        data_df = pd.read_excel("data/loan_default_data.xlsx")
        return data_df
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}") from e

def transform_term(text):
    try:
        new_text = float(text.lower().replace("months","").strip())
        return new_text
    except:
        return -1


def transform_employment_length(text):
    try:
        new_text = text.lower()
        new_text = "".join([s for s in new_text if s == "<" or s.isdigit()])
        new_text = new_text.replace("<1", "0.5")
        new_text = float(new_text)
        return new_text
    except:
        return 0


def transform_home_ownership(text):
    # The values where set based on failure rate that was calculated within 01_data_analysis.py
    dict = {"own" : 15.58,
            "rent" : 15.83,
            "mortgage" : 14.28,
            "other" : 23.20}
    if text.lower() in dict.keys():
        return dict[text.lower()]
    else:
        # any other value will set to as "other" home ownership
        return 23.20


def transform_verification_status(text):
    try:
        # The value assignment are based on the arrangement from least to worst failure rate
        dict = {"source verified" : 14.77,
                "verified" : 16.80}
        if text.lower() in dict.keys():
            return dict[text.lower()]
        else:
            # any other value will set to as "Not Verified" home ownership
            return 14.17
    except:
        # any other value will set to as "Not Verified" home ownership
        return 14.17


def transform_date(date_text):
    try:
        return date_text.timestamp()
    except:
        return 0


def transform_zip_code(zip_code):
    try:
        new_zip_code = str(zip_code)[:3]
        return float(new_zip_code)
    except:
        return 0


def transform_revolving_utillization(text):
    try:
        new_text = "".join([s for s in text.lower() if s == "." or s.isdigit()])
        return float(new_text)
    except:
        return 0


def transform_loan_status(loan_status):
    try:
        status_value = None
        # sorted from the best to worst status
        dict = {"Fully Paid": 6,
                "Current": 5,
                "In Grace Period": 4,
                "Late (16-30 days)": 3,
                "Late (31-120 days)": 2,
                "Charged Off": 1,
                "Default": 0
                }
        for stats in dict.keys():
            if stats.lower() in loan_status.lower():
                status_value = dict[stats]
                break
        return status_value
    except:
        return 0.5


def transform_purpose(purpose):
    # The values where set based on failure rate that was calculated within 01_data_analysis.py
    try:
        dict = {"major_purchase": 10.49,
                "other": 17.22,
                "debt_consolidation": 15.52,
                "credit_card": 11.58,
                "small_business": 27.82,
                "medical": 17.33,
                "wedding": 11,
                "car": 11.01,
                "home_improvement": 13.24,
                "educational": 21.24,
                "vacation": 14.72,
                "house": 16.28,
                "moving": 16.55,
                "renewable_energy": 17.58}
        return dict[purpose.lower()]
    except:
        # any other values will be treated as "other" purpose
        return 17.22

def transform_address_state(state):
    """
    This function treat the address state such as "AL" as a base 24 digit.
    The symbol A represents the value 1, B represents 2, C represents 3, and so on.
    The function intends to convert these 'base 24' digit into base 10 digit.
    Note: Base 10 digits are the ones we are using daily.
    """
    try:
        state = str(state).upper()
        base26_value = 0
        for char in state:
            # in ASCII, A is 65, B is 66, C is 67 ...
            # the "ord(char) - 64" returns the position of the char
            base26_value = base26_value * 26 + ord(char) - 64
        # value '0' will be reserved for unknown, absent value, etc. hence, address state of "A" is 1 instead of 0.
        base26_value += 1
        return base26_value
    except:
        return 0


def transform_months_since_last_delinquency(months_since_last_delinquency):
    try:
        if not math.isnan(float(months_since_last_delinquency)):
            return float(months_since_last_delinquency)
        else:
            return 0
    except:
        return 0


def transform_nan_num(value):
    try:
        if not math.isnan(float(value)):
            return float(value)
        else:
            return -1
    except:
        return -1


def data_transformation(df):
    new_df = pd.DataFrame()
    transform_function = {"loan_amount": transform_nan_num,
                          "funded_amount": transform_nan_num,
                          "funded_amount_investors": transform_nan_num,
                          "term" : transform_term,
                          "interest_rate": transform_nan_num,
                          "installment": transform_nan_num,
                          "employment_length" : transform_employment_length,
                          "home_ownership": transform_home_ownership,
                          "annual_income": transform_nan_num,
                          "verification_status": transform_verification_status,
                          "issue_date": transform_date,
                          "loan_status": transform_loan_status,
                          "purpose": transform_purpose,
                          "zip_code": transform_zip_code,
                          "address_state": transform_address_state,
                          "debt_to_income_ratio": transform_nan_num,
                          "no_delinquency_2yrs": transform_nan_num,
                          "earliest_credit_line": transform_date,
                          "inquiries_last_6mths": transform_nan_num,
                          "months_since_last_delinquency": transform_months_since_last_delinquency,
                          "no_open_accounts": transform_nan_num,
                          "public_records": transform_nan_num,
                          "revolving_balance": transform_nan_num,
                          "revolving_utillization": transform_revolving_utillization,
                          "no_total_account": transform_nan_num,
                          "total_payment": transform_nan_num,
                          "total_payment_investors": transform_nan_num,
                          "total_received_principal": transform_nan_num,
                          "total_received_interest": transform_nan_num,
                          "last_payment_date": transform_date,
                          "last_payment_amnt": transform_nan_num,
                          "next_payment_date": transform_date,
                          "last_credit_pull_date": transform_date,
                          "meet_credit_policy": transform_nan_num,
                          "purpose_risk": transform_nan_num,
                          "purpose_essential": transform_nan_num,
                          "exist_months_since_last_delinquency": transform_nan_num}

    for column_name in df.keys():
        if column_name in transform_function:
            new_df[column_name] = df[column_name].apply(transform_function[column_name])
        else:
            # no transformation done on columns that are not defined in transform_function variable
            new_df[column_name] = df[column_name]

    return new_df


def extract_meet_credit_policy(loan_status):
    try:
        loan_status = loan_status.lower()
        if "does not meet the credit policy" in loan_status:
            return 0
        else:
            return 1
    except:
        return 0.5


def extract_asset_type(purpose):
    """
        - 1 represents Assets (possible investment)
        - 0.5 represents ambiguous, unknown or not sure.
        - 0 represents Expenses
    """
    try:
        dict = {"major_purchase" : 0.5,
                "other" : 0.5,
                "debt_consolidation" : 0,
                "credit_card" : 0,
                "small_business" : 1,
                "medical" : 0,
                "wedding" : 0,
                "car" : 1,
                "home_improvement" : 0.5,
                "educational" : 1,
                "vacation" : 0,
                "house" : 1,
                "moving" : 0,
                "renewable_energy": 1}
        return dict[purpose.lower()]
    except:
        return 0.5


def extract_essential_type(purpose):
    """
        - 1 represents essential spending
        - 0.5 represents ambiguous, unknown or not sure.
        - 0 represents nonessential spending
    """
    try:
        dict = {"major_purchase" : 0.5,
                "other" : 0.5,
                "debt_consolidation" : 0,
                "credit_card" : 0,
                "small_business" : 1,
                "medical" : 1,
                "wedding" : 0.5,
                "car" : 0.5,
                "home_improvement" : 0,
                "educational" : 0.5,
                "vacation" : 0,
                "house" : 1,
                "moving" : 0,
                "renewable_energy": 0}
        return dict[purpose.lower()]
    except:
        return 0.5


def extract_exist_months_since_last_delinquency(months_since_last_delinquency):
    """
        - 1 represents existence of last delinquency
        - 0.5 represents unknown, for what ever reason if failed to identify 0 or 1
        - 0 represents no delinquency
    """
    try:
        if math.isnan(float(months_since_last_delinquency)):
            return 0
        else:
            return 1
    except:
        return 0.5


def column_extraction(df):
    extraction_function = {"meet_credit_policy": [extract_meet_credit_policy, "loan_status"],
                           "purpose_risk": [extract_asset_type,"purpose"],
                           "purpose_essential": [extract_essential_type,"purpose"],
                           "exist_months_since_last_delinquency": [extract_exist_months_since_last_delinquency,"months_since_last_delinquency"]}

    for new_column_name in extraction_function.keys():
        function = extraction_function[new_column_name][0]
        column_name = extraction_function[new_column_name][1]
        if column_name in df.keys():
            df[new_column_name] = df[column_name].apply(function)
    return df


def normalise(final_df):
    normalised_df = pd.DataFrame()
    scaler = MinMaxScaler()
    return normalised_df


def prepare_data(raw_data) -> pd.DataFrame:
    final_df = column_extraction(raw_data)
    final_df = data_transformation(final_df)
    final_df.to_csv("data/loan_cleaned_data.csv")
    return final_df
    """final_df = normalise(final_df)
    for column_name in normalise_list:
        new_data_df[column_name] = scaler.fit_transform(new_data_df[[column_name]])
        # Save the scaler to a file
        with open(f'config/model/{column_name}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    return new_data_df"""