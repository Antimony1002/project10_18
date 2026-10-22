# Deployment boundary

The trained ACT policy produces one 14-D vector in this order:

```text
left_joint_0..5, left_gripper, right_joint_0..5, right_gripper
```

`action_adapter.py` validates and splits that vector. It intentionally does **not** send commands to the
robot. Before implementing the sender, confirm the I2RT inference API, joint limits, command rate, emergency
stop behavior, and whether the teleoperation leader values require an extra calibration transform before they
can be used as follower targets.

Do not connect a checkpoint to the physical robot until an offline replay shows that the predicted ranges and
ordering match the recorded leader commands.
