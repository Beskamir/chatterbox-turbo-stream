import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import time

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)


# Generate with Paralinguistic Tags
text = "Oh, that's hilarious! [chuckle]"

# Record the start time
start_time = time.perf_counter()

# text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)

# Record the end time
end_time = time.perf_counter()

ta.save("test-1.wav", wav, model.sr)

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
# text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
# wav = multilingual_model.generate(text, language_id="fr")
# ta.save("test-2.wav", wav, multilingual_model.sr)


# # If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# ta.save("test-3.wav", wav, model.sr)
