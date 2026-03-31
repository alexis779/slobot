from slobot.sim.golf_ball_env import GolfBallEnv

ball_x = -5
ball_y = -11
cup_x = 8
cup_y = -11

golf_ball_env = GolfBallEnv()
golf_ball_env.set_object_initial_positions(ball_x, ball_y, cup_x, cup_y)

observation = ...
action = jepa_policy.action(observation)
golf_ball_env.arm.genesis.entity.control_dofs_position(action)