from processes.train_test import get_train_test_data, train_neural_network, train_k_nearest_neighbor

if __name__=="__main__":
    try:
        train_df, test_df = get_train_test_data()
        model = train_k_nearest_neighbor(train_df, test_df)

        print("Neural network trained successfully!")
    except Exception as e:
        print("Unexpected error occurred during training and testing")