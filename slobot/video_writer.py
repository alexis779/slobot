from multiprocessing import shared_memory
import numpy as np
import subprocess

class VideoWriter():
    FFMPEG_BINARY = "ffmpeg"

    def __init__(self, filename, res, fps, codec):
        self.filename = filename
        self.res = res
        self.fps = fps
        self.codec = codec

    def transcode(self, raw_video, output_filename):
        shared_memory = self.shared_memory(raw_video)
        shared_memory_filename = f"/dev/shm/{shared_memory.name}"

        cmd = [
            VideoWriter.FFMPEG_BINARY,
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (self.res[0], self.res[1]),
            "-r", "%.02f" % self.fps,
            "-an",
            "-i", shared_memory_filename,
            "-vcodec", self.codec,
            output_filename
        ]

        # execute cmd
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        process.wait()
        if process.returncode != 0:
            error_message = process.stderr.read().decode()
            process.stderr.close()
            print(f"FFmpeg error: {error_message}")
            raise RuntimeError(f"FFmpeg failed with error code {process.returncode}")

        shared_memory.unlink()


    def shared_memory(self, raw_video):
        shm = shared_memory.SharedMemory(create=True, size=raw_video.nbytes)
        shm_raw_video = np.ndarray(raw_video.shape, dtype=raw_video.dtype, buffer=shm.buf)
        shm_raw_video[:] = raw_video[:]
        shm.close()
        return shm
    