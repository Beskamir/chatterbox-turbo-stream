# Description

David Browne's chatterbox-streaming was a little bit too slow on my 2080 so I modified resemble-ai's chatterbox-turbo model to support streaming. Besides, the non-turbo version doesn't appear to have support for paralinguistic tags, and that was a feature I really wanted to have for the eventual project that I want to make with this text to speech program. Regardless, I got this library to a somewhat decent state, and it's likely others with older hardware like mine will find it useful.

---

# Instructions

## Windows Installation

Developed and tested with python version 3.11 so it's recommended you use that version too.

Install torch depedancy with cuda:
```shell
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```
Other versions may work better, but this is what I used.
Torchvision is probably entirely unnecessary.
Torchaudio might have issues with ffmpeg. It's why I switched to soundfile in the audio_saver class, but there are still uses of torchaudio elsewhere in this project and those will likely be very broken.

(Potentially optional since I switched to soundfile)
Install ffmpeg:
```shell
winget install ffmpeg
```
May need experimental version of ffmpeg or older version of torch.


Finally install this library from source:
```shell
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
py -3.11 -m pip install -e .
```

## Usage

Refer to example_stream_tts_turbo.py for a better example but in short:
```python
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.audio_player import ContinuousAudioPlayer
from chatterbox.audio_saver import AudioSaver
# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Sample text with Paralinguistic Tags
text = "[laugh] Hello there! Here are some example input tags and text for the model to speak with the streamed version of Chatterbox's turbo model. Meaning it'll generate audio while simultaneously playing what it has already generated. So instead of waiting for this entire text to generate before you hear anything, you can start listening to this text within only a couple seconds."

# Init and start the real time, continuous audio player
audioPlayer = ContinuousAudioPlayer(sample_rate=model.sr)
audioPlayer.start()

# Init the audio saver
audioSaver = AudioSaver(filepath="output-streamer.wav", sample_rate=model.sr)

# Loop that generates audio and adds it to the player/saver
for audio_chunk in model.generate_stream(
    text=text,
    chunk_size=50,  # tokens per chunk
    context_window=100,
    temperature=0.9,
):
    # Add audio to audio player
    audioPlayer.add_audio(audio_chunk)
    # Add audio to audio saver
    audioSaver.add(audio_chunk)

# Save the audio file
audioSaver.save()
# Stops the audio player:
audioPlayer.stop()
```

## Development and Debugging 

Search `Debug Note` in the repository to find optional print/debug sections I removed for performance reasons.

Main files I modified or added are chatterbox/audio_player.py, chatterbox/audio_saver.py, chatterbox/tts_turbo.py, chatterbox/models/t3/t3.py, and example_stream_tts_turbo.py

Also changed the S3GEN_SR constant in chatterbox/models/s3gen/const.py since 24000 seemed too fast while 23000 was much nicer, and removed LoRACompatibleLinear from chatterbox/models/s3gen/matcha/transformer.py since it was giving me a deprecation warning, but I didn't get around to replacing it with PEFT, so if fine tuning is important to you then go look into how to do that properly. For me a simple nn.Linear was sufficient. This is likely more of a training than inference thing anyway so probably fine to just leave it like this.

---

# Performance

## Laptop RTX3070ti

Generating Audio Chunks:
Generated audio chunk in 1.72 seconds
Generated audio chunk in 1.06 seconds
Generated audio chunk in 1.02 seconds
Generated audio chunk in 0.90 seconds
Generated audio chunk in 1.02 seconds
Generated audio chunk in 0.88 seconds
Generated audio chunk in 0.89 seconds
Generated audio chunk in 0.94 seconds
**Total generating time 8.43 seconds, but latency to first audio chunk is 1.72s.**

Generating audio with Chatterbox's normal turbo generator:
**Generates entire audio without streaming in 6.28 seconds, and only then plays the generated audio.**

## Desktop RTX2080

Generating Audio Chunks:
Generated audio chunk in 2.42 seconds
Generated audio chunk in 1.68 seconds
Generated audio chunk in 1.77 seconds
Generated audio chunk in 1.70 seconds
Generated audio chunk in 1.72 seconds
Generated audio chunk in 1.70 seconds
Generated audio chunk in 1.72 seconds
Generated audio chunk in 1.68 seconds
Generated audio chunk in 1.47 seconds
**Total generating time: 15.85 seconds, but latency to first audio chunk is 2.42s.**

Saved streaming audio to: output-streamer.wav
Total streaming chunks: 9
Final audio shape: torch.Size([1, 426240])
Final audio duration: 18.532s

Generating audio with Chatterbox's normal turbo generator:
**Generates entire audio without streaming in 13.75 seconds, and only then plays the generated audio.**

Saved streaming audio to: output-normal.wav
Total streaming chunks: 1
Final audio shape: torch.Size([1, 447360])
Final audio duration: 19.450s

---

# TODO

- The blending between the audio chunks isn't quite perfect yet.
- Potentially a better way of supporting multiple voices than a dictionary.
- Whispering as either a post processing effect or from audio samples.
- Properly update/fix the deprecated LoRACompatibleLinear feature I removed.
- Add any other optimizations I can think of or come across.
- Improve audio quality as much as possible.

---

# Acknowledgements
- [Chatterbox-TTS](https://github.com/resemble-ai/chatterbox)
- [chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming)
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

---

# Disclaimer
- Please don't use this model to do bad things.
- Prompts are sourced from freely available data on the internet.
