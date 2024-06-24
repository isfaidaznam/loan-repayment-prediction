from config.config import Config
import pandas as pd
from processes.formating import print_title


def dataframe_resampling(df, sample_size):
    # Sort by total_received_principal, then perform systematic sampling
    df_sorted = df.sort_values(by='total_received_principal')
    sample_interval = len(df_sorted) // sample_size
    sampled_indices = []
    for i in range(0, len(df_sorted), sample_interval):
        sampled_indices.append(i)
    sampled_df = df_sorted.iloc[sampled_indices].reset_index(drop=True)
    return sampled_df


def prepare_train_test_data(data_df):
    try:
        config = Config()
        train_test_split = config.ML_TRAINING['TRAIN_SPLIT']

        # Split fail and pass
        fail_df = data_df[data_df["repay_fail"] == max(data_df["repay_fail"])]
        pass_df = data_df[data_df["repay_fail"] == min(data_df["repay_fail"])]

        # Randomise order
        fail_df = fail_df.sample(frac=1, random_state=7).reset_index(drop=True)
        pass_df = pass_df.sample(frac=1, random_state=7).reset_index(drop=True)

        # Get index to separate train and test
        split_index_fail = int(len(fail_df) * train_test_split / 100)
        split_index_pass = int(len(pass_df) * train_test_split / 100)

        # Get training data
        train_fail_df = fail_df.head(min(split_index_fail, split_index_pass))
        train_pass_df = pass_df.head(min(split_index_fail, split_index_pass))

        # By remove all index from train dataframe, this will get Test data
        # These were not trimmed due to imbalance data to reflect on real world use
        test_fail_df = fail_df.tail(len(fail_df) - split_index_fail)
        test_pass_df = pass_df.tail(len(pass_df) - split_index_pass)

        train_df = pd.concat([train_fail_df, train_pass_df])
        test_df = pd.concat([test_fail_df, test_pass_df])

        # Randomise order for performance of training
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)

        train_df.to_csv('data/train_data.csv', index=False, header=True)
        test_df.to_csv('data/test_data.csv', index=False, header=True)

        print("Train and test data saved as csv.")

        # Distribution details
        print_title("Class Distribution of Repayment Failure within Train & Test Data")
        data = [
            [train_df["repay_fail"].value_counts()[-1],
             test_df["repay_fail"].value_counts()[-1]],
            [train_df["repay_fail"].value_counts()[1],
             test_df["repay_fail"].value_counts()[1]]
        ]
        distribution_df = pd.DataFrame(data, index=['repay_fail = 0', 'repay_fail = 1'], columns=['train data', 'test data'])
        print(distribution_df)
    except Exception as e:
        print(f"Unexpected error during test and train data preparation: {e}")