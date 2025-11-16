## Examples

### Real

This example moves the robot to the preset positions, waiting 1 sec in between each one.

```
python scripts/real.py
```

<video controls src="https://github.com/user-attachments/assets/54d11b46-accf-499b-97ac-ce53533c1029"></video>

### Sim To Real

This example performs the 3 elemental rotations in sim and real.
The simulation generates steps, propagating the joint positions to the Feetech motors.

```
python scripts/sim_to_real.py
```

| sim                                                                                                            | real                                                                                                           |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| <video controls src="https://github.com/user-attachments/assets/986acdbf-e6da-4101-a319-4b04179d96b4"></video> | <video controls src="https://github.com/user-attachments/assets/dad3467b-d474-4b33-8910-9c86f88b3784"></video> |

### Image stream

Genesis camera provides access to each frames rendered by the rasterizer. Multiple types of image are provided:

- RGB
- Depth
- Segmentation
- Surface

The following script iterates through all the frames, calculating the FPS metric every second.

```
python scripts/sim_fps.py
...
FPS= FpsMetric(1743573645.3103304, 0.10412893176772242)
FPS= FpsMetric(1743573646.3160942, 59.656155690238116)
FPS= FpsMetric(1743573647.321373, 59.68493363485116)
FPS= FpsMetric(1743573649.8052156, 12.078059963768446)
FPS= FpsMetric(1743573650.8105915, 59.67917299445178)
FPS= FpsMetric(1743573651.8152244, 59.723304924655935)
...
```

<video controls src="https://github.com/user-attachments/assets/37c518c8-6a4a-4a66-b450-f8b1f0eb052d"></video>
