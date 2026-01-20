import time
import threading
import queue
from collections import deque
from .models.s3gen import S3GEN_SR

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
    print("Using sounddevice for audio playback")
except ImportError:
    AUDIO_AVAILABLE = False
    print("sounddevice not available. Install with: pip install sounddevice")


class ContinuousAudioPlayer:
    """
        Audio player for streaming audio chunks in real time

        sample_rate: The audio's sample rate
        block_size: The audio buffer's size
    """
    def __init__(self, sample_rate: int = S3GEN_SR, block_size: int = 1024):
        self.sample_rate = sample_rate
        self.blocksize = block_size

        # Queue of numpy float32 arrays
        self._buffer = deque()
        self._lock = threading.Lock()

        self._stream = None
        self._playing = False

    # Threaded method that will run in background
    def start(self):
        """Start audio playback (call once)."""
        if (not AUDIO_AVAILABLE) or (self._playing):
            return

        def audio_callback(outdata, frames, time_info, status):
            # Always initialize output (prevents garbage)
            outdata.fill(0)

            with self._lock:
                i = 0
                while i < frames and self._buffer:
                    chunk = self._buffer[0]

                    # Number of samples we can take from this chunk
                    take = min(len(chunk), frames - i)

                    outdata[i:i + take, 0] = chunk[:take]

                    # Shrink or drop the chunk
                    if take < len(chunk):
                        self._buffer[0] = chunk[take:]
                    else:
                        self._buffer.popleft()

                    i += take

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=audio_callback,
        )

        self._stream.start()
        self._playing = True

    # Add an audio chunk to the buffer to be played as soon as possible
    def add_audio(self, audio_chunk):
        """Append a torch or numpy audio chunk to the playback buffer."""
        if not self._playing:
            return

        audio_np = audio_chunk.squeeze().numpy().astype(np.float32)

        if audio_np.size == 0:
            return

        with self._lock:

            # Append the chunk
            self._buffer.append(audio_np)

    # Wait for any audio to finish playing, then stop the player
    def stop(self):
        """Stop playback after buffer drains."""
        if self._stream and self._playing:
            # Let remaining audio play out
            while True:
                with self._lock:
                    if not self._buffer:
                        break
                time.sleep(0.01)

            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._playing = False
            self._fade_tail = None  # clear fade tail