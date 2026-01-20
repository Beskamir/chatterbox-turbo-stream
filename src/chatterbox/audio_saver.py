import torch
import soundfile as sf
from .models.s3gen import S3GEN_SR

class AudioSaver():
    """
        This class handles saving audio to a file.
        filepath: where/what you want to save the audio as
        sample_rate: The audio's sample rate
    """
    def __init__(self, filepath="output.wav", sample_rate: int = S3GEN_SR):
        self._path = filepath
        self._audio_chunks = []
        self._sample_rate = sample_rate

    """
        Add an audio chunk to be saved in the output file
        audio_chunk: The audio to be added to the output file
    """
    def add(self, audio_chunk):
        self._audio_chunks.append(audio_chunk)

    """
        save the audio to a file
    """
    def save(self):

        if(self._audio_chunks):
            full_streamed_audio = torch.cat(self._audio_chunks, dim=-1)

            # Convert to numpy array
            audio_np = full_streamed_audio.cpu().numpy()

            # Transpose to [samples, channels] if multi-channel
            if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
                audio_np = audio_np.T
            elif audio_np.ndim == 1:
                audio_np = audio_np.reshape(-1, 1)  # mono

            # SoundFile expects float32 or int data
            audio_np = audio_np.astype('float32')

            sf.write(self._path, audio_np, self._sample_rate)
            print(f"Saved streaming audio to: " + str(self._path))
            print(f"Total streaming chunks: {len(self._audio_chunks)}")
            print(f"Final audio shape: {full_streamed_audio.shape}")
            print(f"Final audio duration: {full_streamed_audio.shape[-1] / self._sample_rate:.3f}s")

        else:
            print("Error: Lacking audio to save!")