import os
import math
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Generator, Tuple, Optional

import librosa
import torch
# import perth
import pyloudnorm as ln

from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL
import numpy as np

import torchaudio.functional as F
# import logging
# logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0

class ChatterboxTurboTTS:
    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    voices = {}

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self._prev_chunk = None
        # self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTurboTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        # Turbo specific hp
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        s3gen.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTurboTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=False,
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device)

    def norm_loudness(self, wav, sr, target_lufs=-27):
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            print(f"Warning: Error in norm_loudness, skipping: {e}")

        return wav

    def audio_sample(self, wav_fpath, exaggeration=0.0, norm_loudness=True):
        """
            Method for supplying a voice sample. Must be 5 seconds or longer.
        """
        if(wav_fpath not in self.voices):
            ## Load and norm reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

            assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

            if norm_loudness:
                s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

            # Speech cond prompt tokens
            if plen := self.t3.hp.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
            # self.conds = Conditionals(t3_cond, s3gen_ref_dict)
            self.voices[wav_fpath] = Conditionals(t3_cond, s3gen_ref_dict)

        self.conds = self.voices[wav_fpath]
        self._prev_chunk = None

    def generate(
        self,
        text,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
    ):
        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(self.device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )

        return wav.detach().cpu()
        
    def _process_token_buffer(
        self,
        token_buffer,
        all_tokens_so_far,
        context_window,
        fade_duration=0.002,
        crossfade_duration=0.004,
        zero_crossing_range=2,
    ):
        """
            Helper method for converting tokens into audio
            token_buffer: The select tokens to be converted into audio
            all_tokens_so_far: All the tokens generated so far
            context_window: How many already generated tokens should be included
            fade_duration: Seconds to apply linear fade in & out on each chunk
            crossfade_duration, Seconds to apply crossfade between chunks
            zero_crossing_range, Seconds to search for a zero crossing
        """
        # Combine buffered chunks of tokens
        new_tokens = torch.cat(token_buffer, dim=-1)

        # Build tokens_to_process by including a context window
        if len(all_tokens_so_far) > 0:
            context_tokens = (
                all_tokens_so_far[-context_window:]
                if len(all_tokens_so_far) > context_window
                else all_tokens_so_far
            )
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0

        # Drop any invalid tokens and move to the correct device
        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.device)
        if len(clean_tokens) == 0:
            return None, False

        # Run S3Gen inference to get a waveform (1 × T)
        wav, _ = self.s3gen.inference(
            speech_tokens=clean_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2
        )

        wav = wav.squeeze(0).cpu()

        # If we have context tokens, crop out the samples corresponding to them
        if context_length > 0:
            samples_per_token = wav.shape[-1] // clean_tokens.shape[-1]
            skip_samples = context_length * samples_per_token
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if audio_chunk.shape[-1] == 0:
            return None, False

        audio_chunk = audio_chunk.clone()
        fade_len = int(fade_duration * self.sr)
        fade_in = torch.linspace(0.0, 1.0, fade_len)
        fade_out = torch.linspace(1.0, 0.0, fade_len)
        audio_chunk[:fade_len] *= fade_in
        audio_chunk[-fade_len:] *= fade_out

        # --- ZERO CROSSING STITCHING ---
        # If we have a previous chunk, find zero crossing near boundary
        if self._prev_chunk is not None:
            prev_chunk = self._prev_chunk
            # Look at last N samples of previous chunk + first N samples of new chunk
            max_search = min(len(prev_chunk), len(audio_chunk), int(self.sr * zero_crossing_range))
            boundary_region = torch.cat([prev_chunk[-max_search:], audio_chunk[:max_search]])

            # Find first zero-crossing in new chunk portion
            zero_crossings = torch.where(boundary_region[:-1] * boundary_region[1:] <= 0)[0]
            zero_crossings = zero_crossings[zero_crossings >= max_search]  # only in new chunk
            if len(zero_crossings) > 0:
                audio_chunk = audio_chunk[zero_crossings[0]-max_search:]  # trim to zero crossing

            # Optional: minimal crossfade to avoid tiny discontinuities
            crossfade_samples = min(int(self.sr * crossfade_duration), len(prev_chunk), len(audio_chunk))
            if crossfade_samples > 0:
                fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
                fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
                audio_chunk[:crossfade_samples] = audio_chunk[:crossfade_samples] * fade_in + prev_chunk[-crossfade_samples:] * fade_out

        # Save current chunk for next iteration
        self._prev_chunk = audio_chunk.clone()

        audio_tensor = audio_chunk.unsqueeze(0)

        return audio_tensor, True

    def generate_stream(
        self,
        text: str,
        chunk_size: int = 50,
        context_window: int = 500,
        fade_duration: float=0.002,
        crossfade_duration: float=0.004,
        zero_crossing_range: float=2.0,
        temperature: float = 0.8,
        top_k: float=1000,
        top_p: float=0.95,
        repetition_penalty: float=1.2,
        max_new_tokens: int=1000
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming version of generate that yields audio chunks as they are generated.
        
        Args:
            text: Input text to synthesize
            chunk_size: Number of speech tokens per chunk
            context_window: How many already generated tokens should be included
            fade_duration: Seconds to apply linear fade in & out on each chunk
            crossfade_duration, Seconds to apply crossfade between chunks
            zero_crossing_range, Seconds to search for a zero crossing
            temperature: Controls randomness (larger: random, smaller: deterministic)
            top_k: keeps most likely tokens (larger: unstable, smaller: robotic)
            top_p: keeps smallest set of tokens with probability > p
            repetition_penalty: Penalizes tokens that were already generated
            max_new_tokens: This seems like it might be necessary to avoid issues
        Yields:
            audio_chunk is a torch.Tensor
        """
        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        # Presumably this pads text with start of text and end of text tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=sot)
        text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=eot)

        all_tokens_processed = []  # Keep track of all tokens processed so far
        
        for token_chunk in self.t3.inference_turbo_stream(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k = top_k,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
        ):
            # Extract only the conditional batch
            token_chunk = token_chunk[0]
                
            # Process each chunk immediately
            audio_tensor, success = self._process_token_buffer([token_chunk], all_tokens_processed, context_window, fade_duration, crossfade_duration, zero_crossing_range)

            if success:
                yield audio_tensor

            # Update all_tokens_processed with the new tokens
            if len(all_tokens_processed) == 0:
                all_tokens_processed = token_chunk
            else:
                all_tokens_processed = torch.cat([all_tokens_processed, token_chunk], dim=-1)

