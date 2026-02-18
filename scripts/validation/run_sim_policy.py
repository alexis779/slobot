#!/usr/bin/env python3
"""
CLI script to run the golf ball pickup simulation policy.

Usage:
    python run_sim_policy.py --ball-x -5 --ball-y -2 --cup-x -7 --cup-y -13 --pre-grasp-mode (vertical|vertical-flip|horizontal)
"""

import argparse
from slobot.validation.sim_policy import PreGraspMode, SimPolicy

def main():
    parser = argparse.ArgumentParser(
        description='Run golf ball pickup simulation with SoArm100'
    )
    
    parser.add_argument(
        '--ball-x',
        type=float,
        required=True,
        help='Golf ball X position in inches'
    )
    
    parser.add_argument(
        '--ball-y',
        type=float,
        required=True,
        help='Golf ball Y position in inches'
    )
    
    parser.add_argument(
        '--cup-x',
        type=float,
        required=True,
        help='Cup X position in inches'
    )
    
    parser.add_argument(
        '--cup-y',
        type=float,
        required=True,
        help='Cup Y position in inches'
    )
    
    parser.add_argument(
        '--pre-grasp-mode',
        type=str,
        choices=['vertical', 'vertical-flip', 'horizontal'],
        default='horizontal',
        help='Pre-grasp approach mode (default: horizontal)'
    )

    args = parser.parse_args()

    # Convert pre-grasp-mode string to enum
    args.pre_grasp_mode = PreGraspMode(args.pre_grasp_mode)

    # Create and execute policy
    policy = SimPolicy(**vars(args))
    
    # Execute the task
    success = policy.execute()
    
    # Report result
    if success:
        print("✓ Task completed successfully!")
        exit(0)
    else:
        print("✗ Task failed - ball not in cup")
        exit(1)

if __name__ == '__main__':
    main()
