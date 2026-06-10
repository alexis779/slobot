## Camera feed

If you intend to record with your Android phone, install **Webcam IP** Android app on your phone, select 640 x 480 image resolution and start server.

List v4l2 devices

```
v4l2-ctl --list-devices
```

Create a looback device if /dev/video4 is missing in the above output.

```
sudo apt install linux-headers-$(uname -r) v4l2loopback-dkms

sudo modprobe v4l2loopback devices=1 video_nr=4
```

Create a virtual camera via:

```
ffmpeg -i http://192.168.0.102:8080/video -f v4l2 -pix_fmt yuyv422 /dev/video4
```

Make sure the camera is streamable via

```
ffplay /dev/video4
```

Wether it's via phone app or webcam, make sure you're able to capture the camera feed.

```
ffmpeg -f v4l2 -video_size 640x480 -framerate 30 -i /dev/video4 -c:v libx264 episode01.mp4 -y
```

Press q to stop the capture. Replay the recording via

```
ffplay -autoexit episode01.mp4
```

### Webcam configuration

Check the formats available via

```
% v4l2-ctl --list-formats-ext -d /dev/video3
```

Following command dumps the current webcam resolution and FPS.

```
% v4l2-ctl -d /dev/video3 --get-fmt-video
Format Video Capture:
        Width/Height      : 640/480
        Pixel Format      : 'YUYV' (YUYV 4:2:2)
        Field             : None
        Bytes per Line    : 1280
        Size Image        : 614400
        Colorspace        : sRGB
        Transfer Function : Rec. 709
        YCbCr/HSV Encoding: ITU-R 601
        Quantization      : Default (maps to Limited Range)
        Flags             :
```

```
% v4l2-ctl -d /dev/video3 --get-parm
Streaming Parameters Video Capture:
        Capabilities     : timeperframe
        Frames per second: 30.000 (30/1)
        Read buffers     : 0
```

### Night recording risks (auto-exposure, white balance, glare)

At night, USB webcams left on **auto-exposure** and **auto white balance** often produce images that look nothing like daytime recordings of the same scene:

- **Yellow / orange color cast** — AWB locks onto warm lamp light (e.g. 4000 K) instead of neutral grey.
- **Glare and blown highlights** — auto-exposure runs long (e.g. 300+ units vs ~150 on a better-exposed camera), washing out the table and cube.
- **Low SNR** — gain and long exposure add grain; fine cues (gripper–cube gap, cube on target) become hard to learn.
- **Moving banding** — AC mains or PWM-dimmed lamps plus rolling shutter show as faint dark waves on the table.
- **Visible shadows** — side lighting casts gripper shadows toward the camera; overhead lighting removes them.

These artifacts are a problem for **vision-based reward classifiers** trained on human labels: the model sees pixels only (not EE pose). It can memorize night-specific texture and color, then fail on a new session even when the robot pose is almost identical.

#### Example: `so100_cube_rectangle_night` (10 episodes)

Two cameras were recorded (config keys `observation.images.side` and `observation.images.wrist`; device mapping may be swapped vs physical mount — verify with `v4l2-ctl --list-devices`):

| Stream key | Device (this setup) | Night appearance |
|------------|---------------------|------------------|
| `side` | eMeet C950 (`/dev/video2`) | Dark table, grey cube, white jaw — crisp contrast |
| `wrist` | USB camera (`/dev/video4`) | Yellow / orange wash, glare |

**Train** (human `s` labels, `reward=1`):

![Night train success ep0 frame 57 — side (left) and wrist (right)](./images/camera/night_train_success_ep0_f57.png)

**Eval** (classifier should fire, same task, prob ≈ 0.04):

![Night eval miss ep0 frame 57 — side (left) and wrist (right)](./images/camera/night_eval_miss_ep0_f57.png)

Frames look similar to the eye; classifier output collapses on eval. Training reached 100% accuracy on labeled frames; generalization failed.

Additional pairs (success region, frames 56 and 60):

![Night train success ep0 frame 56](./images/camera/night_train_success_ep0_f56.png)

![Night eval miss ep0 frame 60](./images/camera/night_eval_miss_ep0_f60.png)

#### Ablation: which camera hurts the fused classifier?

The reward classifier concatenates embeddings from both cameras, then applies one MLP head. Offline tests on checkpoint `so100_cube_rectangle_night_reward_classifier` (ep0 frame 57):

**Zero ablation** — zero one camera input before inference:

| Eval images | Prob |
|-------------|------|
| Side + wrist (both eval) | 0.001 |
| Side only (wrist zeroed) | **0.95** |
| Wrist only (side zeroed) | 0.003 |

The side stream alone is enough for success; adding the wrist stream collapses the fused score.

**Swap test** — replace one eval camera with the train success image at the same frame index:

| Images used | Prob |
|-------------|------|
| Eval side + eval wrist | 0.001 |
| Train side + eval wrist | 0.36 |
| Eval side + train wrist | 0.56 |
| Train side + train wrist | **0.99** |

Eval wrist hurts more than eval side (larger gain when replaced with train wrist). The only fix is to manually lock camera settings (especially on the glare-heavy USB stream), re-record the dataset, and retrain the reward classifier on **both** cameras.

`v4l2-ctl` profile for `/dev/video4` (USB camera). Each manual value is followed by the driver default.

**Suggested night profile** (reduces yellow cast and glare):

```bash
# --- Exposure (glare) ---
v4l2-ctl -d /dev/video4 -c auto_exposure=1
# default: auto_exposure=3   # Aperture Priority Mode (auto)

v4l2-ctl -d /dev/video4 -c exposure_time_absolute=120
# default: exposure_time_absolute=313   # inactive while auto_exposure=3

# --- White balance (yellow / orange) ---
v4l2-ctl -d /dev/video4 -c white_balance_automatic=0
# default: white_balance_automatic=1   # auto on

v4l2-ctl -d /dev/video4 -c white_balance_temperature=5800
# default: white_balance_temperature=4000   # inactive while white_balance_automatic=1

# --- Color / tone ---
v4l2-ctl -d /dev/video4 -c saturation=40
# default: saturation=51

v4l2-ctl -d /dev/video4 -c contrast=28
# default: contrast=32

v4l2-ctl -d /dev/video4 -c brightness=-10
# default: brightness=0

# --- Optional: hot spots ---
v4l2-ctl -d /dev/video4 -c backlight_compensation=1
# default: backlight_compensation=0

# --- Optional: AC flicker / moving bands on table ---
v4l2-ctl -d /dev/video4 -c power_line_frequency=1
# default: power_line_frequency=1 (50 Hz); use 2 for 60 Hz mains
```

Check current values:

```bash
v4l2-ctl -d /dev/video4 -C auto_exposure,exposure_time_absolute,white_balance_automatic,white_balance_temperature,saturation,contrast,brightness,gamma,gain,backlight_compensation,power_line_frequency
```

**Reset to driver defaults:**

```bash
v4l2-ctl -d /dev/video4 -c auto_exposure=3
v4l2-ctl -d /dev/video4 -c white_balance_automatic=1
v4l2-ctl -d /dev/video4 -c exposure_time_absolute=313
v4l2-ctl -d /dev/video4 -c white_balance_temperature=4000
v4l2-ctl -d /dev/video4 -c saturation=51
v4l2-ctl -d /dev/video4 -c contrast=32
v4l2-ctl -d /dev/video4 -c brightness=0
v4l2-ctl -d /dev/video4 -c gamma=100
v4l2-ctl -d /dev/video4 -c gain=0
v4l2-ctl -d /dev/video4 -c backlight_compensation=0
v4l2-ctl -d /dev/video4 -c power_line_frequency=1
```


## Gripper with camera mount

### 3d print

To install an eye-in-hand camera, replace the [fixed jaw](https://github.com/google-deepmind/mujoco_menagerie/blob/main/trs_so_arm100/assets/Fixed_Jaw.stl) on the SO-ARM-100 robot with [this modified version](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Optional/Wrist_Cam_Mount_32x32_UVC_Module/stl/Wrist_Cam_Mount_32x32_UVC_Module_SO100.stl) that includes a camera mount.

Attach this [camera module](https://www.amazon.com/innomaker-Computer-Raspberry-Support-Windows/dp/B0CNCSFQC1/) to the plate.


### Edit STL

Import the STL with the camera mount in Blender, as well as the STL from the original Mujoco configuration.

Then apply the following transform to `Wrist_Cam_Mount_32x32_UVC_Module_SO100.stl` to match `Fixed_Jaw.stl`.

- Rotation: `(-90, 0, 180)`
- Scale: `(0.001, 0.001, 0.001)`

![Blender Edit](./images/BlenderEdit.png)

Export the modified STL. Then update the visual mesh path of the fixed jaw to the modified STL file.


### Measure screw holes position

In blender, in edit mode, select a vertex on the edge of each screw hold. It will show the coordinates of the vertex.

hole id | location on the plate | 3D position
-|-|-
H1 | bottom left | `(-0.015826, -0.002098, -0.083203)`
H2 | bottom right | `(0.011174, -0.002098, -0.083203)`
H3 | top left | `(-0.015826, 0.00848, -0.060518)`

![Blender Measure 3D Point](./images/BlenderMeasure3DPoint.png)


### Camera extrinsics

We can attach a mobile camera to the Fixed Jaw link in the simulator.

Define the following vectors:
- `u = H1 -> H2`
- `u_n = u / ||u||`
- `v = H1 -> H3`
- `v_n = v / ||v||`
- `w = w_n = u_n x v_n`

Then the camera extrinsics can be roughly estimated by `H1 + (u + v) / 2` as the camera position and `(u_n, v_n, w_n)` as the camera orientation.