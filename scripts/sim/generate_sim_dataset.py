import argparse
from slobot.sim.sim_dataset_generator import SimDatasetGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Run golf ball pickup simulation with SoArm100'
    )
    
    parser.add_argument(
        '--episode-count',
        type=int,
        required=True,
        help='Episode count'
    )

    args = parser.parse_args()

    sim_dataset_generator = SimDatasetGenerator(episode_count=args.episode_count)
    sim_dataset_generator.generate_dataset()

if __name__ == '__main__':
    main()