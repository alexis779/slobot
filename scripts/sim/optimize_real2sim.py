from slobot.sim.real2sim_optimizer import Real2SimOptimizer
import argparse

def main():
    args_parser = argparse.ArgumentParser(description="Minimize the error between the ground truth golf ball position and the sim TCP position at the pick frame.")
    args_parser.add_argument("--input-csv-file", type=str, required=True, help="The file containing the samples.")
    args_parser.add_argument("--output-csv-file", type=str, required=True, help="The file containing the configuration mappings.")
    args = args_parser.parse_args()

    real2sim_optimizer = Real2SimOptimizer()
    real2sim_optimizer.optimize(args.input_csv_file, args.output_csv_file)

if __name__ == "__main__":
    main()