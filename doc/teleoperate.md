# Tele-operation

Run tele-operation, controlling the follower using the leader position.

```
python scripts/teleop/teleoperate.py
```

The robot state for both the leader and the follower can be visualized in rerun.io viewer.

In the example below, the arm was stretched out horizontally to apply the maximal torque possible on the `shoulder_lift` joint. Notice how the follower diverges from the leader. It is not able to lift back the arm to adjust to the leader position. Proportional gain `K_p=16` from the PD controller is insufficient. When multiplied with the angular error, it does not compensate the higher gravity force, causing the arm to sag. Increasing `K_p` gain fixes the error. However the link may start to vibrate at other positions under less load. The PD controller will overshoot the necesary force, causing the joints to oscillate around the target position.

![Gradio dashboard](./TeleopRerun.io.png)



## Gradio app

```
python scripts/teleop/gradio_control_app.py
```