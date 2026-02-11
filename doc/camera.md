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