from slobot.sim.real2sim_linear_regression import Real2SimLinearRegression
import argparse

def main():
    args_parser = argparse.ArgumentParser(description="Calibrate the real2sim configuration mapping.")
    args_parser.add_argument("--input-csv-file", type=str, required=True, help="The file containing the dataset.")
    args_parser.add_argument("--output-csv-file", type=str, required=True, help="The file containing the configuration mappings.")
    args = args_parser.parse_args()

    real2sim_linear_regression = Real2SimLinearRegression()

    real2sim_linear_regression.replay_dataset(args.input_csv_file, args.output_csv_file)

if __name__ == "__main__":
    main()