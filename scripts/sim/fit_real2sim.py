from slobot.sim.real2sim_fit import Real2SimFit
import argparse

def main():
    args_parser = argparse.ArgumentParser(description="Calibrate the real2sim configuration mapping.")
    args_parser.add_argument("--input-csv-file", type=str, required=True, help="The file containing the configuration mapping.")
    args = args_parser.parse_args()

    real2sim_fit = Real2SimFit()
    real2sim_fit.fit(args.input_csv_file)

if __name__ == "__main__":
    main()