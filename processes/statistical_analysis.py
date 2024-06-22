import pandas as pd


def print_categorical_analysis(df, column_name):
    list_of_data = df[column_name]
    counts = list_of_data.value_counts()
    percentages = (counts / len(list_of_data)) * 100
    stats = pd.DataFrame({'Value': counts.index,
                          'Total Instance': counts.values,
                          'Population': percentages.values})
    stats['Population'] = stats['Population'].apply(lambda x: '{:.2f}%'.format(x))
    print(f"{column_name} Statistics:")
    print(stats.to_string(index=False))


def print_stats(df):
    try:
        print("""
=============================================================
Class Distribution of Repayment Failure
=============================================================""")
        print_categorical_analysis(df, "repay_fail")
        print("""
=============================================================
Correlation of Features with Repayment Failure
=============================================================""")
        print(df.corr()["repay_fail"].sort_values(ascending=False))
    except Exception as e:
        print(f"Error displaying Descriptive Analysis: {e}")