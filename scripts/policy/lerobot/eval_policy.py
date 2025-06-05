from slobot.lerobot.policy_evaluator import PolicyEvaluator
import argparse

parser = argparse.ArgumentParser(description="Evaluate a policy for a given robot type and model path")
parser.add_argument(
    "--robot_type",
    type=str,
    help="Type of robot to evaluate"
)
parser.add_argument(
    "--policy_type",
    type=str,
    help="Type of policy to evaluate"
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the trained model"
)

args = parser.parse_args()

if __name__ == "__main__":
    evaluator = PolicyEvaluator(args.robot_type, args.policy_type, args.model_path)
    evaluator.evaluate()