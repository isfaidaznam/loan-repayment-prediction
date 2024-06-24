from processes.data_sampling import prepare_train_test_data
from processes.preprocess_data import read_data, prepare_data
from processes.statistical_analysis import print_stats


if __name__=="__main__":
    try:
        print("Starting Data preparation...")
        print("Accessing Data...")
        raw_data = read_data()
        print("Performing Data preparations...")
        processed_data = prepare_data(raw_data)
        print_stats(processed_data)
    except ValueError as e:
        raise ValueError(f"Unexpected error occurred: {e}") from e