import torch
import time

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.audio_player import ContinuousAudioPlayer
from chatterbox.audio_saver import AudioSaver

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# ------

# Init the turbo text to speech model
model = ChatterboxTurboTTS.from_pretrained(device=device)

# ------

# Give model audio samples of at least 5 seconds long for multiple voices:
# model.audio_sample("AudioSample1.wav")
# model.audio_sample("AudioSample2.wav")

# Then switch back to the first voice at runtime with the following:
# model.audio_sample("AudioSample1.wav")

# ------

# Sample text to test the audio generator:
text = "[laugh] Hello there! Here are some example input tags and text for the model to speak with the streamed version of Chatterbox's turbo model. Meaning it'll generate audio while simultaneously playing what it has already generated. So instead of waiting for this entire text to generate before you hear anything, you can start listening to this text within only a couple seconds."

print("\nGenerating Audio With Audio Streaming:")

# Init and start the real time, continuous audio player
print("Starting Audio Player")
audioPlayer = ContinuousAudioPlayer(sample_rate=model.sr)
audioPlayer.start()

# Init the audio saver
print("Starting Audio Saver")
audioSaver = AudioSaver(filepath="output-streamer.wav", sample_rate=model.sr)

# ------

# Timer used to check performance of generate stream
start_time = time.perf_counter()

elapsed_time = 0

# Loop that generates audio and adds it to the player/saver
print("\nGenerating Audio Chunks:")
for audio_chunk in model.generate_stream(
    text=text,
    chunk_size=50,  # tokens per chunk
    context_window=500,
    temperature=0.9,
):
    gen_time = time.perf_counter() - start_time
    elapsed_time+=gen_time
    print("Generated audio chunk in " + str(gen_time) + " seconds")
    
    # Add audio to audio player
    audioPlayer.add_audio(audio_chunk)

    # Add audio to audio saver
    audioSaver.add(audio_chunk)

    start_time = time.perf_counter()

print("Total generating time: " + str(elapsed_time) + " seconds")

# Save the audio file
audioSaver.save()

print("\nGenerating audio with Chatterbox's normal turbo generator:")
# Remake the audio saver with different filepath.
# Commented out since this is only here for comparison purposes
audioSaver = AudioSaver(filepath="output-normal.wav", sample_rate=model.sr)

# Generates audio with the existing audio generation pipeline
start_time = time.perf_counter()
audio_output = model.generate(text, temperature=0.9)
print("Generated entire audio without streaming in " + str(time.perf_counter() - start_time) + " seconds")

# Plays and saves the audio:
audioPlayer.add_audio(audio_output)
audioSaver.add(audio_output)
audioSaver.save()

# Stops the audio player:
audioPlayer.stop()

