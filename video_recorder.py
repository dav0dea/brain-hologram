import subprocess
import numpy as np

class VideoRecorder:
    def __init__(self, filename, size, fps=30, codec="h264_nvenc", pix_fmt="rgb24"):
        """
        Create a GPU-encoded ffmpeg video writer.

        Args:
            filename (str): Output file path (e.g. 'out.mp4').
            size (tuple[int,int]): (width, height)
            fps (int): frames per second
            codec (str): ffmpeg codec (e.g. 'h264_nvenc', 'libx264')
            pix_fmt (str): pixel format (default 'rgb24')
        """
        self.filename = filename
        self.size = f"{size[0]}x{size[1]}"
        self.fps = fps
        self.codec = codec
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", pix_fmt,
                "-s", self.size,
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-vcodec", codec,
                "-preset", "p4",
                "-loglevel", "error",
                filename,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray):
        """
        Accepts RGB or RGBA frames in [0,1] or [0,255] and composites transparencies to black.
        """
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA image, got shape {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)

        # --- Handle alpha if present ---
        if frame.shape[2] == 4:
            rgb = frame[..., :3].astype(np.float32)
            alpha = frame[..., 3:4].astype(np.float32) / 255.0
            frame = (rgb * alpha + (1 - alpha) * 0).astype(np.uint8)  # black background

        self.proc.stdin.write(frame.tobytes())

    def close(self):
        """Finalize and close the video."""
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait()
        except Exception:
            pass
