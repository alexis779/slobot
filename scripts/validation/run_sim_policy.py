#!/usr/bin/env python3
"""
CLI script to run the golf ball pickup simulation policy.

Usage:
    python run_sim_policy.py --ball_x -5 --ball_y -2 --cup_x -7 --cup_y -13
"""

import argparse
from slobot.validation.sim_policy import SimPolicy

def main():
    parser = argparse.ArgumentParser(
        description='Run golf ball pickup simulation with SoArm100'
    )
    
    parser.add_argument(
        '--ball_x',
        type=float,
        required=True,
        help='Golf ball X position in inches'
    )
    
    parser.add_argument(
        '--ball_y',
        type=float,
        required=True,
        help='Golf ball Y position in inches'
    )
    
    parser.add_argument(
        '--cup_x',
        type=float,
        required=True,
        help='Cup X position in inches'
    )
    
    parser.add_argument(
        '--cup_y',
        type=float,
        required=True,
        help='Cup Y position in inches'
    )
    
    args = parser.parse_args()
    
    # Create and execute policy
    policy = SimPolicy(**vars(args))
    
    # Execute the task
    success = policy.execute()
    #policy.arm.genesis.hold_entity()
    
    # Report result
    if success:
        print("✓ Task completed successfully!")
        exit(0)
    else:
        print("✗ Task failed - ball not in cup")
        exit(1)

if __name__ == '__main__':
    main()
