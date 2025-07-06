from multiprocessing import shared_memory
import numpy as np
import subprocess

class VideoWriter():
    FFMPEG_BINARY = "ffmpeg"

    def __init__(self, res, fps, codec):
        self.res = res
        self.fps = fps
        self.codec = codec

    def transcode_numpy(self, numpy_buffer, output_filename):
        shared_memory = self.shared_memory(numpy_buffer)
        shared_memory_filename = f"/dev/shm/{shared_memory.name}"

        cmd = self.ffmpeg_cmd(shared_memory_filename, output_filename)

        # execute cmd
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        process.wait()
        if process.returncode != 0:
            error_message = process.stderr.read().decode()
            process.stderr.close()
            print(f"FFmpeg error: {error_message}")
            raise RuntimeError(f"FFmpeg failed with error code {process.returncode}")

        shared_memory.unlink()

    def ffmpeg_cmd(self, input_filename, output_filename):
        cmd = [
            VideoWriter.FFMPEG_BINARY,
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (self.res[0], self.res[1]),
            "-r", "%.02f" % self.fps,
            "-an",
            "-i", input_filename,
            "-y"
        ]

        if output_filename.startswith("/dev/video"):
            cmd.append("-f")
            cmd.append("v4l2")
            cmd.append("-codec")
            cmd.append("rawvideo")
        else:
            cmd.append("-codec")
            cmd.append("libx264")

        cmd.append(output_filename)

        return cmd

    def open_stream(self, output_filename):
        """
        Starts an ffmpeg subprocess for piping raw video frames (RGB24) to a file.
        Stores the process and pipe for later use.
        """
        cmd = self.ffmpeg_cmd("-", output_filename)
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.pipe = self.process.stdin

    def close_stream(self):
        self.process.stdin.close()
        self.process.wait()

    def transcode_stream(self, buffer):
        buffer = np.array(buffer) # make sure the data is contiguous in memory
        self.pipe.write(buffer.tobytes())

    def shared_memory(self, numpy_buffer):
        shm = shared_memory.SharedMemory(create=True, size=numpy_buffer.nbytes)
        shm_raw_video = np.ndarray(numpy_buffer.shape, dtype=numpy_buffer.dtype, buffer=shm.buf)
        shm_raw_video[:] = numpy_buffer[:]
        shm.close()
        return shm
    