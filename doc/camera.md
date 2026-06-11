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

### Reward classifier and fixed camera settings

The HIL-SERL reward classifier is **images-only** (side + wrist). It fuses both streams in one head. If the **side camera** keeps **auto-exposure** or **auto white balance** enabled, aperture and color temperature drift between the train recording and a later eval session — even when the robot pose looks identical.

#### Example: `so100_cube_rectangle_day` (episode 0)

The side camera is this [Tewiky Wide Angle Camera](https://www.amazon.com/dp/B08VJ25PL1).

| Stream key | Device | Physical mount |
|------------|--------|----------------|
| `observation.images.side` | `/dev/video2` | eMeet C950 — fixed side view |

Train labels reached 100% classifier accuracy. On eval (`so100_cube_rectangle_day_eval` ep0), the classifier never crossed the 0.5 threshold despite a successful place. Side-camera train vs eval frames look the same to the eye; mean pixel difference is only **5.8 / 255 (~2.3%)**, concentrated in **table glare** (top-right) and a slight global warmth shift — enough to collapse the fused score when auto settings differ across sessions.

![Side camera train f105 vs eval f53 — original frames, diff ×8, hotspots](./images/camera/side_train_vs_eval_diff_panel.png)

Panel columns: **train f105** | **eval f53** | **diff ×8** (per-channel \|eval − train\| × 8 as RGB) | **hotspots** (red where any channel differs by > 12/255). Yellow/white speckle in diff ×8 is glare instability, not gripper motion.

#### Ablation: which camera hurts the fused classifier?

The reward classifier concatenates embeddings from both cameras, then applies one MLP head. Offline tests on checkpoint `so100_cube_rectangle_day_reward_classifier` (eval ep0 frame 53):

| Stream key | Device |
|------------|--------|
| `observation.images.side` | `/dev/video2` (eMeet, side) |
| `observation.images.wrist` | `/dev/video4` (USB, wrist) |

**Zero ablation** — zero one camera input before inference:

| Eval images | Prob |
|-------------|------|
| Side + wrist (both eval) | 0.001 |
| Side only (wrist zeroed) | 0.94 |
| Wrist only (side zeroed) | 0.02 |

Fused eval side + eval wrist collapses to 0.001. Side-only reaches 0.94 — the side view looks like success in isolation, but the eval side stream does not match train in the fused head. Wrist-only stays low (0.02); the side camera is what drifts between sessions.

**Swap test** — replace one eval camera with the train success image at the same frame index:

| Images used | Prob |
|-------------|------|
| Eval side + eval wrist | 0.001 |
| Train side + eval wrist | 0.15 |
| Eval side + train wrist | 0.96 |
| Train side + train wrist | **0.99** |

Replacing eval **side** with train side raises 0.001 → 0.15; train side + train wrist reaches 0.99. The side camera on `/dev/video2` (auto exposure / white balance drift, table glare) is the stream that collapses the fused score — lock its settings and re-record before retraining.

**Fix:** lock exposure and white balance on `/dev/video2` (side) before every train and eval recording, using the workflow below.

#### Offline replay vs recorded live probability

`next.reward.probability` in the dataset is computed on the **live** preprocessed tensors at record time. Offline replay re-decodes frames from the dataset's **lossy AV1 videos**. Encode/decode changes pixels (blocking, color shift, glare smoothing) enough to move classifier scores — even when frame indices line up. That is the main reason decoded offline replay diverges from the recorded live series, not async writer misalignment.

The reward classifier is sensitive to small visual deltas (see the ~2% side-camera drift above). A lossy round-trip through AV1 can flip a near-threshold frame.

**Record path:** `slobot-record` stores live resized tensors in parquet as `observation.preprocessed_images.*` (float32 `[C,H,W]`, no AV1 round-trip) and encodes `observation.images.*` as AV1 video for viz / policy training. Train the reward classifier with `slobot-train`, which reads the preprocessed parquet columns.

#### Workflow: dump, disable auto, hardcode

Check the current settings

```bash
% v4l2-ctl -d /dev/video2 -C white_balance_automatic,white_balance_temperature,auto_exposure,exposure_time_absolute
white_balance_automatic: 1
white_balance_temperature: 4600
auto_exposure: 3 (Aperture Priority Mode)
exposure_time_absolute: 157
```

Disable auto settings
```bash
v4l2-ctl -d /dev/video2 -c auto_exposure=1
v4l2-ctl -d /dev/video2 -c white_balance_automatic=0
```

See effect:

<video controls src="https://github.com/user-attachments/assets/d2084311-d788-4533-9d2b-5049301b42b0"></video>

The settings should now look like

```bash
$ v4l2-ctl -d /dev/video2 -C white_balance_automatic,white_balance_temperature,auto_exposure,exposure_time_absolute
white_balance_automatic: 0
white_balance_temperature: 4600
auto_exposure: 1 (Manual Mode)
exposure_time_absolute: 157
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