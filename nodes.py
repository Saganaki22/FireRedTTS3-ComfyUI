"""ComfyUI node definitions for FireRedTTS3."""

from __future__ import annotations

import logging

import torch

from . import native
from .loader import (
    ATTENTION_OPTIONS,
    DEVICE_OPTIONS,
    DTYPE_OPTIONS,
    VARIANTS,
    get_repo_choices,
    load_firered_bundle,
    resume_bundle_to_device,
)
from .tokenizer import LANGUAGE_CHOICES
from .whisper import FireRedWhisperTranscribe

logger = logging.getLogger("FireRedTTS3")

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

PROGRESS_UNITS_PER_SENTENCE = 1000
CATEGORY = "FireRedTTS3"


def _text_input(default: str, tooltip: str) -> tuple:
    return ("STRING", {"multiline": True, "default": default, "tooltip": tooltip})


def _generation_controls(default_cfg: float) -> dict:
    return {
        "n_timesteps": (
            "INT",
            {
                "default": 10,
                "min": 1,
                "max": 50,
                "step": 1,
                "tooltip": "Flow-matching steps per generated audio patch. 10 is the official default; more is slower with diminishing returns.",
            },
        ),
        "inference_cfg": (
            "FLOAT",
            {
                "default": default_cfg,
                "min": 0.0,
                "max": 4.0,
                "step": 0.05,
                "tooltip": "Classifier-free guidance strength for the flow head. 0 disables CFG. Official defaults: 2.0 for cloning, 1.2 for design/edits.",
            },
        ),
        "stop_threshold": (
            "FLOAT",
            {
                "default": 0.5,
                "min": 0.05,
                "max": 0.95,
                "step": 0.05,
                "tooltip": "Stop-token probability threshold that ends generation. Higher values allow longer audio.",
            },
        ),
        "seed": (
            "INT",
            {
                "default": 42,
                "min": 0,
                "max": 2**31 - 1,
                "tooltip": "0 uses the current random state. A positive value is repeatable.",
            },
        ),
        "max_audio_seconds": (
            "FLOAT",
            {
                "default": 64.0,
                "min": 4.0,
                "max": 160.0,
                "step": 1.0,
                "tooltip": "Hard cap on generated audio length per sentence (64s is the official maximum).",
            },
        ),
    }


def _frontend_controls() -> dict:
    return {
        "do_tn": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "Run text normalization (numbers, dates, units to spoken form). Chinese/English use local wetext; other languages get basic cleaning.",
            },
        ),
        "do_split": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "Split long text into sentences and generate them one by one (cross-faded together).",
            },
        ),
        "cross_fade_ms": (
            "FLOAT",
            {
                "default": 50.0,
                "min": 0.0,
                "max": 500.0,
                "step": 10.0,
                "tooltip": "Cross-fade between sentence segments in milliseconds.",
            },
        ),
    }


def _max_gen_steps(max_audio_seconds: float) -> int:
    # 25 latent frames per second, 4 latents generated per AR step.
    return max(6, int(round(float(max_audio_seconds) * 25.0 / 4.0)))


def _concat_segments(segments: list[torch.Tensor], sample_rate: int, cross_fade_ms: float) -> dict:
    if not segments:
        raise RuntimeError("No audio segments were generated.")
    gen_audio = segments[0]
    if len(segments) > 1:
        fade_len = int(cross_fade_ms / 1000.0 * sample_rate)
        for segment in segments[1:]:
            gen_audio = native.cross_fade(gen_audio, segment, fade_len)
    if not torch.isfinite(gen_audio).all():
        raise RuntimeError("FireRedTTS3 generated non-finite audio samples.")
    return native.tensor_audio_to_comfy(gen_audio, sample_rate)


def _generate_clone_audio(
    bundle,
    *,
    text: str,
    language: str,
    prompt_text: str,
    prompt_audio: dict,
    n_timesteps: int,
    inference_cfg: float,
    stop_threshold: float,
    seed: int,
    max_audio_seconds: float,
    do_tn: bool,
    do_split: bool,
    cross_fade_ms: float,
) -> dict:
    resume_bundle_to_device(bundle)
    waveform, sample_rate = native.comfy_audio_to_tensor(prompt_audio)
    text, language, sentences = bundle.frontend.apply(
        text,
        language=None if language == "auto" else language,
        do_tn=bool(do_tn),
        do_split=bool(do_split),
        tokenize=lambda s: native.measure_tokens(bundle, s),
    )
    logger.info("%s cloning: %d sentence(s), language=%s", bundle.variant, len(sentences), language)
    if bundle.variant != "base" and language not in ("Chinese", "English", "Cantonese"):
        logger.warning(
            "FireRedTTS3-Instruct cloning is unreliable for %s (verified: non-ZH/EN fails even with "
            "a same-language reference). Switch the loader variant to fireredtts3_base for multilingual "
            "voice cloning.", language)

    prompt_latents, prompt_audio_len = native.tokenize_prompt_audio(bundle, waveform, sample_rate)
    spk_emb = None
    if bundle.variant == "base":
        spk_emb = native.speaker_embedding(bundle, waveform, sample_rate)

    max_gen_steps = _max_gen_steps(max_audio_seconds)
    progress_total = len(sentences) * PROGRESS_UNITS_PER_SENTENCE
    pbar = ProgressBar(progress_total) if ProgressBar is not None else None
    segments: list[torch.Tensor] = []
    gen_audio_sr = None
    for index, sentence in enumerate(sentences):
        logger.info("sentence %d/%d: %s", index + 1, len(sentences), sentence[:60])

        def update(current: int, total: int, sentence_index: int = index) -> None:
            if pbar is None:
                return
            fraction = min(1.0, max(0.0, float(current) / max(1, int(total))))
            pbar.update_absolute(sentence_index * PROGRESS_UNITS_PER_SENTENCE + round(fraction * PROGRESS_UNITS_PER_SENTENCE), progress_total)

        if bundle.variant == "base":
            segment, gen_audio_sr = native.base_clone_one(
                bundle,
                text=sentence,
                language=language,
                prompt_text=prompt_text,
                prompt_latents=prompt_latents,
                prompt_audio_len=prompt_audio_len,
                spk_emb=spk_emb,
                stop_threshold=stop_threshold,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed,
                max_gen_steps=max_gen_steps,
                progress_callback=update,
            )
        else:
            segment, gen_audio_sr = native.instruct_clone_one(
                bundle,
                text=sentence,
                prompt_text=prompt_text,
                prompt_latents=prompt_latents,
                stop_threshold=stop_threshold,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed,
                max_gen_steps=max_gen_steps,
                progress_callback=update,
            )
        segments.append(segment.cpu())
        if pbar is not None:
            pbar.update_absolute((index + 1) * PROGRESS_UNITS_PER_SENTENCE, progress_total)
    return _concat_segments(segments, gen_audio_sr, cross_fade_ms)


def _require_instruct(bundle, node_name: str) -> None:
    if bundle.variant != "instruct":
        raise RuntimeError(
            f"{node_name} needs the Instruct model, but the FireRedTTS3 Load Model node has "
            f"variant='fireredtts3_base' loaded (from {bundle.model_dir.name}). "
            f"Fix: set variant to 'fireredtts3_instruct' on the FireRedTTS3 Load Model node and re-run. "
            f"Voice Clone works with both variants; Voice Design, Semantic Edit and Acoustic Edit are Instruct-only."
        )


class FireRedTTS3LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo": (
                    get_repo_choices(),
                    {
                        "default": "FireRedTTS3-bf16",
                        "tooltip": "Weight source. bf16 keeps the flow head/decoder in fp32 and matches official mixed-precision compute (recommended). int8 ConvRot quantizes the transformer linears (smallest, experimental). fp32 is the official full-precision repo. Missing files download when download_if_missing is on; otherwise the error tells you where to place them.",
                    },
                ),
                "variant": (
                    VARIANTS,
                    {
                        "default": "fireredtts3_instruct",
                        "tooltip": "base: zero-shot cloning in 24 languages + 21 Chinese dialects using language tags. instruct: voice design (Chinese/English only) + speech editing; its cloning inherits the language from the reference audio (no language tags).",
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "bf16 stores the backbone LLM and RedAE encoder in bf16 (same compute as the official autocast path) and keeps the flow head/decoder fp32. fp32 is full precision. auto picks bf16 on supported GPUs.",
                    },
                ),
                "device": (
                    DEVICE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Device for inference. auto follows ComfyUI's current torch device; cpu is a slow fallback.",
                    },
                ),
                "attention": (
                    ATTENTION_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Attention backend for the Qwen3 transformers. auto uses flash_attention when flash_attn is installed and compatible (CUDA + bf16 compute), else sdpa. The fp32 RedAE decoder always uses sdpa.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download the selected weights, RedAE codec, tokenizer, CAM++ and FastText language-ID files into ComfyUI/models/fireredtts3 when missing.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FIREREDTTS3_MODEL",)
    RETURN_NAMES = ("firered_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load FireRedTTS3 (base or instruct) natively with ComfyUI/AIMDO memory registration."

    def load(self, repo: str, variant: str, dtype: str, device: str, attention: str, download_if_missing: bool):
        bundle = load_firered_bundle(
            repo_choice=repo,
            variant=variant,
            dtype_name=dtype,
            device_name=device,
            attention=attention,
            download_if_missing=bool(download_if_missing),
        )
        return (bundle,)


class FireRedTTS3VoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "firered_model": ("FIREREDTTS3_MODEL",),
            "text": _text_input(
                "Hello! This is FireRedTTS3 running natively inside ComfyUI.",
                "Text to synthesize. Long text is split into sentences automatically when do_split is on.",
            ),
            "prompt_audio": (
                "AUDIO",
                {"tooltip": "Reference voice clip for zero-shot cloning. Clean speech with little noise works best."},
            ),
            "prompt_text": _text_input(
                "",
                "Exact transcript of the reference clip. Strongly improves cloning quality.",
            ),
            "language": (
                LANGUAGE_CHOICES,
                {
                    "default": "auto",
                    "tooltip": "Language or Chinese dialect tag. auto uses FastText (24 languages) with zh/ja/en heuristic fallback. For best cloning, match the prompt audio language.",
                },
            ),
        }
        required.update(_generation_controls(default_cfg=2.0))
        required.update(_frontend_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Zero-shot voice cloning with FireRedTTS3 base or instruct."

    def generate(self, firered_model, text, prompt_audio, prompt_text, language, n_timesteps, inference_cfg,
                 stop_threshold, seed, max_audio_seconds, do_tn, do_split, cross_fade_ms):
        audio = _generate_clone_audio(
            firered_model,
            text=text,
            language=language,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            stop_threshold=stop_threshold,
            seed=seed,
            max_audio_seconds=max_audio_seconds,
            do_tn=do_tn,
            do_split=do_split,
            cross_fade_ms=cross_fade_ms,
        )
        return (audio,)


class FireRedTTS3VoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "firered_model": ("FIREREDTTS3_MODEL",),
            "instruction": _text_input(
                "A gentle young female voice, speaking a little slowly, with a playful touch.",
                "Natural-language voice description (gender, age, timbre, emotion, pace, accent). Any language works; the model card's Chinese examples are also good templates. No reference audio needed.",
            ),
            "text": _text_input(
                "Hello! This is FireRedTTS3 running natively inside ComfyUI.",
                "Text to synthesize with the designed voice.",
            ),
            "language": (
                ["auto", "Chinese", "English"],
                {"default": "auto", "tooltip": "Language of the text; used for text normalization and sentence splitting only. Voice design synthesis itself only works reliably in Chinese/English (upstream limitation: no language conditioning without a reference clip); other languages produce accented/garbled speech. For other languages, design the voice in English/Chinese, then clone the result with the base model and a language tag."},
            ),
            "text_temperature": (
                "FLOAT",
                {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                 "tooltip": "Sampling temperature for the model's voice-plan text (Chain-of-Thought)."},
            ),
            "text_top_p": (
                "FLOAT",
                {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Top-p for the voice-plan text."},
            ),
            "text_top_k": (
                "INT",
                {"default": 20, "min": 0, "max": 500, "step": 1, "tooltip": "Top-k for the voice-plan text. 0 disables."},
            ),
            "text_repetition_penalty": (
                "FLOAT",
                {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Repetition penalty for the voice-plan text. 1.0 is the official default."},
            ),
        }
        required.update(_generation_controls(default_cfg=1.2))
        required.update(_frontend_controls())
        del required["stop_threshold"]
        return {"required": required}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "voice_plan")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Design a brand-new voice from a text description (FireRedTTS3-Instruct)."

    def generate(self, firered_model, instruction, text, language, text_temperature, text_top_p, text_top_k,
                 text_repetition_penalty, n_timesteps, inference_cfg, seed, max_audio_seconds,
                 do_tn, do_split, cross_fade_ms):
        _require_instruct(firered_model, "Voice Design")
        resume_bundle_to_device(firered_model)
        text, _language, sentences = firered_model.frontend.apply(
            text,
            language=None if language == "auto" else language,
            do_tn=bool(do_tn),
            do_split=bool(do_split),
            tokenize=lambda s: native.measure_tokens(firered_model, s),
        )
        max_gen_steps = _max_gen_steps(max_audio_seconds)
        progress_total = len(sentences) * PROGRESS_UNITS_PER_SENTENCE
        pbar = ProgressBar(progress_total) if ProgressBar is not None else None
        segments: list[torch.Tensor] = []
        gen_audio_sr = None
        voice_plan = ""
        for index, sentence in enumerate(sentences):
            logger.info("voice design sentence %d/%d: %s", index + 1, len(sentences), sentence[:60])

            def update(current: int, total: int, sentence_index: int = index) -> None:
                if pbar is None:
                    return
                fraction = min(1.0, max(0.0, float(current) / max(1, int(total))))
                pbar.update_absolute(sentence_index * PROGRESS_UNITS_PER_SENTENCE + round(fraction * PROGRESS_UNITS_PER_SENTENCE), progress_total)

            segment, gen_audio_sr, segment_plan = native.voice_design_one(
                firered_model,
                instruction=instruction,
                text=sentence,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                text_repetition_penalty=text_repetition_penalty,
                max_gen_steps=max_gen_steps,
                progress_callback=update,
            )
            segments.append(segment.cpu())
            if index == 0:
                voice_plan = segment_plan
            if pbar is not None:
                pbar.update_absolute((index + 1) * PROGRESS_UNITS_PER_SENTENCE, progress_total)
        logger.info("voice plan: %s", voice_plan[:150])
        return (_concat_segments(segments, gen_audio_sr, cross_fade_ms), voice_plan)


class FireRedTTS3SemanticEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "firered_model": ("FIREREDTTS3_MODEL",),
                "audio": ("AUDIO", {"tooltip": "Input speech to edit."}),
                "instruction": _text_input(
                    "Replace 'cats' with 'dogs'.",
                    "Content edit instruction: insertion, deletion or substitution, e.g. \"insert 'really' after the word at index 8.\"",
                ),
                **_generation_controls(default_cfg=1.2),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "edited_text")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Semantic speech editing (insert/delete/replace words) with FireRedTTS3-Instruct."

    def generate(self, firered_model, audio, instruction, n_timesteps, inference_cfg, stop_threshold,
                 seed, max_audio_seconds):
        _require_instruct(firered_model, "Semantic Edit")
        resume_bundle_to_device(firered_model)
        waveform, sample_rate = native.comfy_audio_to_tensor(audio)
        latents_in, _padded = native.tokenize_prompt_audio(firered_model, waveform, sample_rate)
        max_gen_steps = _max_gen_steps(max_audio_seconds)
        pbar = ProgressBar(max_gen_steps) if ProgressBar is not None else None

        def update(current: int, total: int) -> None:
            if pbar is not None:
                pbar.update_absolute(min(current, max_gen_steps), max_gen_steps)

        segment, gen_audio_sr, edited_text = native.semantic_edit_one(
            firered_model,
            instruction=instruction,
            latents_in=latents_in,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            seed=seed,
            stop_threshold=stop_threshold,
            max_gen_steps=max_gen_steps,
            progress_callback=update,
        )
        if not torch.isfinite(segment).all():
            raise RuntimeError("FireRedTTS3 generated non-finite audio samples.")
        logger.info("semantic edit text: %s", edited_text[:150])
        return (native.tensor_audio_to_comfy(segment, gen_audio_sr), edited_text)


def _acoustic_instruction(mode: str, value: float) -> str:
    if mode == "speed":
        return f"adjust the speed to {value:.1f}x"
    if mode == "volume":
        return f"adjust the volume to {value:.1f}x"
    steps = int(round(value))
    return f"shift the pitch by {steps} step" + ("s" if abs(steps) != 1 else "")


class FireRedTTS3AcousticEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "firered_model": ("FIREREDTTS3_MODEL",),
                "audio": ("AUDIO", {"tooltip": "Input speech to transform."}),
                "mode": (
                    ["speed", "pitch", "volume"],
                    {"default": "speed", "tooltip": "Acoustic attribute to edit. Uses the model's trained instruction templates."},
                ),
                "value": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": -6.0,
                        "max": 6.0,
                        "step": 0.1,
                        "tooltip": "speed: 0.5-2.0 (rate multiplier). volume: 0.3-2.0 (gain multiplier). pitch: -6 to +6 semitone-like steps (rounded to an integer, not 0).",
                    },
                ),
                **_generation_controls(default_cfg=1.2),
            },
            "optional": {
                "custom_instruction": _text_input(
                    "",
                    "Optional raw instruction override (e.g. 'adjust the speed to 0.5x'). Leave empty to build it from mode + value.",
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Acoustic speech editing (speed / pitch / volume) with FireRedTTS3-Instruct."

    def generate(self, firered_model, audio, mode, value, n_timesteps, inference_cfg, stop_threshold,
                 seed, max_audio_seconds, custom_instruction=""):
        _require_instruct(firered_model, "Acoustic Edit")
        instruction = custom_instruction.strip() or _acoustic_instruction(mode, float(value))
        resume_bundle_to_device(firered_model)
        waveform, sample_rate = native.comfy_audio_to_tensor(audio)
        latents_in, _padded = native.tokenize_prompt_audio(firered_model, waveform, sample_rate)
        max_gen_steps = _max_gen_steps(max_audio_seconds)
        pbar = ProgressBar(max_gen_steps) if ProgressBar is not None else None

        def update(current: int, total: int) -> None:
            if pbar is not None:
                pbar.update_absolute(min(current, max_gen_steps), max_gen_steps)

        logger.info("acoustic edit: %s", instruction)
        segment, gen_audio_sr = native.acoustic_edit_one(
            firered_model,
            instruction=instruction,
            latents_in=latents_in,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            seed=seed,
            stop_threshold=stop_threshold,
            max_gen_steps=max_gen_steps,
            progress_callback=update,
        )
        if not torch.isfinite(segment).all():
            raise RuntimeError("FireRedTTS3 generated non-finite audio samples.")
        return (native.tensor_audio_to_comfy(segment, gen_audio_sr),)


NODE_CLASS_MAPPINGS = {
    "FireRedTTS3LoadModel": FireRedTTS3LoadModel,
    "FireRedTTS3VoiceClone": FireRedTTS3VoiceClone,
    "FireRedTTS3VoiceDesign": FireRedTTS3VoiceDesign,
    "FireRedTTS3SemanticEdit": FireRedTTS3SemanticEdit,
    "FireRedTTS3AcousticEdit": FireRedTTS3AcousticEdit,
    "FireRedTTS3WhisperTranscribe": FireRedWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FireRedTTS3LoadModel": "FireRedTTS3 Load Model",
    "FireRedTTS3VoiceClone": "FireRedTTS3 Voice Clone",
    "FireRedTTS3VoiceDesign": "FireRedTTS3 Voice Design",
    "FireRedTTS3SemanticEdit": "FireRedTTS3 Semantic Edit",
    "FireRedTTS3AcousticEdit": "FireRedTTS3 Acoustic Edit",
    "FireRedTTS3WhisperTranscribe": "FireRedTTS3 Whisper Transcribe",
}
