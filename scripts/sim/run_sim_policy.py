import argparse
from slobot.sim.sim_policy import SimPolicy
from slobot.sim.golf_ball_env import GolfBallEnv
from slobot.sim.recording_layout import PreGraspMode, RecordingLayout

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
        help='Pre-grasp approach mode'
    )

    args = parser.parse_args()

    golf_ball_env = GolfBallEnv()

    # Create and execute policy
    sim_policy = SimPolicy(golf_ball_env)

    # Convert pre-grasp-mode string to enum
    pre_grasp_mode = PreGraspMode(args.pre_grasp_mode)

    recording_layout = RecordingLayout(rrd_file=None, pre_pick_frame_id=None, pick_frame_id=None, pre_grasp_mode=pre_grasp_mode, ball_x=args.ball_x, ball_y=args.ball_y, cup_x=args.cup_x, cup_y=args.cup_y)

    # Execute the task
    success = sim_policy.execute(recording_layout)
    
    # Report result
    if success:
        print("✓ Task completed successfully!")
        exit(0)
    else:
        print("✗ Task failed - ball not in cup")
        exit(1)

if __name__ == '__main__':
    main()
