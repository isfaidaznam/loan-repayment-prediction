from processes.data_visualisation import generate_data_visual
from processes.preprocess_data import read_data


if __name__=="__main__":
    try:
        print("Starting Data analysis...")
        print("Accessing Data...")
        raw_data = read_data()
        print("Generating Data Visuals...")
        generate_data_visual(raw_data)
        print("Data analysis completed.")
    except ValueError as e:
        raise ValueError(f"Unexpected error occurred: {e}") from e