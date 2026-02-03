## Acknowledgements

### The Robot Studio

[SO-ARM-100](https://github.com/TheRobotStudio/SO-ARM100) provides CAD & [STL model](https://github.com/TheRobotStudio/SO-ARM100/blob/main/stl_files_for_3dprinting/Follower/Print_Follower_SO_ARM100_08k_Ender.STL) of the robotic arm links. After 3D-printing them and ordering the remaining parts, the robot can be assembled for prototyping.

### Feetech

[Feetech STS3115](https://www.feetechrc.com/74v-19-kgcm-plastic-case-metal-tooth-magnetic-code-double-axis-ttl-series-steering-gear.html) is the servo inserted in each joint of the robot.

- It's _motor_ rotates the connected link.
- It's _magnetic encoder_ measures the absolute angular position of the joint.

### LeRobot

[LeRobot](https://github.com/huggingface/lerobot) provides SOTA policies to perform _Imitation Learning_ and _Reinforcement Learning_ in robotics.

Tele-operate to create a new dataset: have the follower arm perform a task of interest, by replicating the motion of the leader arm held a human operator.

Then fine-tune a model on the training dataset.

Finally evaluate it on the eval dataset to see how well it performs.

```
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

### Genesis

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) is the physics engine running the simulation.

```
@software{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Universal and Generative Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
```

### Rerun.io

[rerun.io](https://rerun.io/) provides a data logging and visualization tool extremely handy for live preview of an episode. The episode recording can be stored locally and replayed at a later time. The multi-modal dataset can be also be queried for reporting and analysis.