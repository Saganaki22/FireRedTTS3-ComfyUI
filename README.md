# FireRedTTS3-ComfyUI

<img width="1040" height="1058" alt="Screenshot 2026-08-14 205748" src="https://github.com/user-attachments/assets/9d1c3470-5329-4d4d-a6cd-a02f2a36d51a" />


**English** | **[中文](./README_zh.md)**

**Version: v0.2.5**

ComfyUI nodes for [FireRedTeam/FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3): zero-shot voice cloning across 24 languages and 21 Chinese dialects, instruction-based voice design, semantic + acoustic speech editing, Whisper reference transcription, and ComfyUI/AIMDO DynamicVRAM support.

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FireRedTeam%2FFireRedTTS3-blue)](https://huggingface.co/FireRedTeam/FireRedTTS3)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-drbaph%2FFireRedTTS3--bf16-green)](https://huggingface.co/drbaph/FireRedTTS3-bf16)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Saganaki22/FireRedTTS3-ComfyUI/blob/main/LICENSE)

> License note: FireRedTTS3 is released by the FireRed Team under Apache-2.0 for academic research purposes. Do not use voice cloning without consent.

## Features

- **Native in-process inference** - The Qwen3 backbone, PatchEncoder, DiT flow head, RedAE codec, and CAM++ speaker encoder run directly inside ComfyUI; no external server, no remote code.
- **Two model variants** - `fireredtts3_base` (zero-shot cloning with language tags) and `fireredtts3_instruct` (cloning + voice design + speech editing), sharing one RedAE codec.
- **24 languages + 21 Chinese dialects** - Explicit language/dialect dropdown, or `auto` with FastText lid.176 detection (heuristic zh/ja/en fallback).
- **Voice design** - Describe a voice in natural language (gender, age, timbre, pace, accent) and synthesize it; the model's voice plan (CoT) is returned as text.
- **Speech editing** - Semantic editing (insert/delete/replace words, returns the rewritten text) and acoustic editing (speed 0.5-2.0x, pitch ±6 steps, volume 0.3-2.0x) via the trained instruction templates.
- **Text frontend included** - Local `wetext` normalization for Chinese/English (numbers, dates, units to spoken form), sentence splitting with cross-fade joins. No external API calls.
- **ComfyUI AUDIO in/out** - Reference voices, edit inputs, and generated audio all use standard ComfyUI `AUDIO`.
- **Optimized flow stepping** - timestep embeddings, per-block AdaLN modulations, and rotary frequencies are computed once per generation instead of once per patch, and the generation buffers are preallocated: ~11-16% faster with bit-identical output (verified by same-seed A/B).
- **AIMDO DynamicVRAM support** - Core, codec, and speaker encoder are registered as separate ComfyUI models with castable weights, paging through ComfyUI/AIMDO when DynamicVRAM is active.
- **bf16 mirror** - Optional half-size weights with official-equivalent mixed precision: the backbone LLM and RedAE encoder are stored in bf16 (the official code already computes them under bf16 autocast), while the flow head and RedAE decoder stay fp32. Same-seed A/B testing against the official fp32 weights produced identical waveforms (cosine 1.0000, SNR > 80 dB).
- **Whisper transcription node** - Turns a reference clip into the `prompt_text` transcript that noticeably improves cloning.
- **INT8 ConvRot weights (experimental)** - the `FireRedTTS3-int8` mirror ships the core transformer linears pre-quantized to Comfy INT8 ConvRot (format `int8_tensorwise`, per-row fp32 scales, offline Hadamard weight rotation, group size 256) using the official comfy-kitchen quantizer. The loader auto-detects `*.comfy_quant` keys and executes through `comfy_kitchen.int8_linear(convrot=True)` with online activation rotation; everything else (embeddings, boundary projections, stop_head, Conv1d, RedAE, CAM++) stays float. 321/332 linears -> checkpoint 7.90 -> 3.07 GiB, with validated output quality.
- **No keep-loaded toggle, no unload node** - The loader handles model-switch cleanup internally.

## Installation

### Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/FireRedTTS3-ComfyUI.git
cd FireRedTTS3-ComfyUI
python install.py
```

For this local Windows setup:

```powershell
...\venv\Scripts\python.exe ...\ComfyUI\custom_nodes\FireRedTTS3-ComfyUI\install.py
```

Restart ComfyUI after installing or updating.

`install.py` works with both **pip** and **uv**, and never installs, upgrades, or removes `torch`, `torchaudio`, `transformers`, or `numpy` - it only adds small torch-free packages (`huggingface_hub`, `safetensors`, `tokenizers`, `accelerate`, `regex`, `tqdm`, and optionally `wetext` + `fasttext-predict`).

## Transformers Compatibility

This nodepack is built for Transformers **5.3.0+** (tested on 5.3.0). It loads FireRedTTS3 natively instead of relying on a remote-code `AutoModel` path: the Qwen3 backbone, PatchEncoder, DiT flow head, and RedAE modules are constructed directly, and the safetensors weights are mapped in by their original key names.

## Language Support by Feature

| Feature | Language support |
| --- | --- |
| **Voice Clone (base)** | 24 languages + 21 Chinese dialects via explicit tag, or `auto` (FastText detection) |
| **Voice Clone (instruct)** | Reliable for Chinese and English. Cantonese is usable with a same-language reference but may degrade with cross-language references. Other languages fail even with a native-language reference (verified: Greek reference + Greek text collapsed) - use the base model for multilingual cloning. |
| **Voice Design (instruct)** | Chinese and English only - an upstream limitation: with no reference audio and no language tag, the model has nothing to condition language on (its own benchmark only evaluates ZH/EN). For other languages, design the voice in English/Chinese, then clone that output with the base model and a language tag. |
| **Semantic / Acoustic Edit (instruct)** | Inherits the input audio's language (verified: Greek audio edited at 1.5x speed keeps its Greek transcript). |

## Model Files

Weights are stored per source repo under `ComfyUI/models/fireredtts3/` (primary folder first; extra_model_paths.yaml entries and symlinks are searched too). Missing files download automatically when `download_if_missing` is on, otherwise the error names the expected folder.

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 fireredtts3/
        ├── 📂 drbaph_FireRedTTS3-bf16/        (bf16 mirror, default)
        │   ├── 📂 fireredtts3_base/
        │   │   ├── config.json
        │   │   └── model.safetensors
        │   ├── 📂 fireredtts3_instruct/
        │   │   ├── config.json
        │   │   └── model.safetensors
        │   ├── 📂 redae/
        │   │   ├── config.json
        │   │   └── model.safetensors
        │   ├── 📂 campp/
        │   │   └── campplus_voxceleb.bin
        │   └── 📂 text_tokenizer/
        │       ├── tokenizer.json
        │       ├── tokenizer_config.json
        │       └── vocab.json
        ├── 📂 drbaph_FireRedTTS3-int8/        (int8 ConvRot mirror, same layout)
        ├── 📂 FireRedTeam_FireRedTTS3/        (official fp32, same layout)
        └── 📂 fasttext/
            └── lid.176.ftz                    (shared language-ID model)
```

Only the selected variant is downloaded; both variants share `redae/`, `campp/`, and `text_tokenizer/`.

| Component | Official fp32 | bf16 mirror | int8 ConvRot mirror |
| --- | --- | --- | --- |
| `fireredtts3_base` | 8.48 GB | 4.70 GiB | 3.30 GB |
| `fireredtts3_instruct` | 8.48 GB | 4.69 GiB | 3.30 GB |
| `redae` | 3.78 GB | 2.46 GiB | unchanged |
| `campp` + tokenizer + fasttext | ~45 MB | ~45 MB | ~45 MB |

Expect roughly **8-14 GB VRAM** depending on variant/dtype and attention backend; AIMDO DynamicVRAM pages castable weights to keep live VRAM pressure low alongside other models.

Manual installs: place the files under `ComfyUI/models/fireredtts3/drbaph_FireRedTTS3-bf16/` (or the matching repo folder) and the loader uses them without downloading.

## Nodes

<details>
<summary><strong>1. FireRedTTS3 Load Model</strong> - Load a FireRedTTS3 variant</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo` | COMBO | `FireRedTTS3-bf16` | Weight source: `FireRedTTS3-bf16` (recommended), `FireRedTTS3-int8` (smallest, experimental), `FireRedTTS3-fp32` (official). Missing files download when `download_if_missing` is on; otherwise the error names the expected folder. |
| `variant` | COMBO | `fireredtts3_instruct` | `fireredtts3_base` (cloning + language tags) or `fireredtts3_instruct` (cloning + design + editing). |
| `dtype` | COMBO | `auto` | `auto`, `bf16`, `fp32`. bf16 stores backbone LLM + RedAE encoder in bf16 and keeps flow head/decoder fp32 (official mixed precision). |
| `device` | COMBO | `auto` | `auto`, `cuda`, `cpu`. `auto` follows ComfyUI's current torch device. |
| `attention` | COMBO | `auto` | `auto` uses `flash_attention` when flash_attn is installed and compatible (CUDA + bf16 compute), else `sdpa`; also explicit `sdpa` / `flash_attention` / `sageattention`. The fp32 RedAE decoder always uses sdpa. |
| `download_if_missing` | BOOLEAN | `True` | Download the selected weights, codec, tokenizer, CAM++, and FastText files if missing. |

**Output:** `firered_model` (`FIREREDTTS3_MODEL`)

</details>

<details>
<summary><strong>2. FireRedTTS3 Voice Clone</strong> - Zero-shot cloning (Base or Instruct)</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `firered_model` | FIREREDTTS3_MODEL | required | Output from Load Model. |
| `text` | STRING | example text | Text to synthesize in the reference voice. |
| `prompt_audio` | AUDIO | required | Clean reference voice clip. |
| `prompt_text` | STRING | empty | Exact transcript of the reference clip. Strongly recommended; right-click the node to convert it to an input and connect Whisper. |
| `language` | COMBO | `auto` | 24 languages + 21 Chinese dialects, or auto-detect. Match the prompt audio language for best cloning. |
| `n_timesteps` | INT | `10` | Flow-matching steps per audio patch. Official default. |
| `inference_cfg` | FLOAT | `2.0` | Classifier-free guidance for the flow head. `0` disables. |
| `stop_threshold` | FLOAT | `0.5` | Stop-token probability that ends generation. |
| `seed` | INT | `42` | `0` is random; a positive value is repeatable. |
| `max_audio_seconds` | FLOAT | `64.0` | Hard cap per sentence (64s is the official maximum). |
| `do_tn` | BOOLEAN | `True` | Text normalization (wetext for zh/en). |
| `do_split` | BOOLEAN | `True` | Split long text into sentences, cross-faded together. |
| `cross_fade_ms` | FLOAT | `50.0` | Cross-fade between sentence segments. |

**Output:** `audio` (`AUDIO`)

</details>

<details>
<summary><strong>3. FireRedTTS3 Voice Design</strong> - Create a voice from a description (Instruct)</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `firered_model` | FIREREDTTS3_MODEL | required | Instruct model from Load Model. |
| `instruction` | STRING | example | Voice description: gender, age, timbre, emotion, pace, accent. |
| `text` | STRING | example text | Text to speak with the designed voice. |
| `language` | COMBO | `auto` | Used for text normalization only. |
| `text_temperature` | FLOAT | `0.7` | Sampling for the model's voice-plan text. |
| `text_top_p` / `text_top_k` / `text_repetition_penalty` | FLOAT/INT/FLOAT | `0.8` / `20` / `1.0` | Voice-plan sampling controls. |
| generation controls | same as clone | | `inference_cfg` defaults to `1.2` here. |

**Outputs:** `audio` (`AUDIO`), `voice_plan` (`STRING`) - the model's voice-attribute plan.

</details>

<details>
<summary><strong>4. FireRedTTS3 Semantic Edit</strong> - Insert / delete / replace words (Instruct)</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `firered_model` | FIREREDTTS3_MODEL | required | Instruct model from Load Model. |
| `audio` | AUDIO | required | Input speech to edit. |
| `instruction` | STRING | example | e.g. `Replace 'cats' with 'dogs'.` or `insert 'really' after the word at index 8.` |
| generation controls | same as clone | | `inference_cfg` defaults to `1.2` here. |

**Outputs:** `audio` (`AUDIO`), `edited_text` (`STRING`) - the model's rewritten transcript.

</details>

<details>
<summary><strong>5. FireRedTTS3 Acoustic Edit</strong> - Speed / pitch / volume (Instruct)</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `firered_model` | FIREREDTTS3_MODEL | required | Instruct model from Load Model. |
| `audio` | AUDIO | required | Input speech to transform. |
| `mode` | COMBO | `speed` | `speed` (0.5-2.0x), `pitch` (±6 steps), or `volume` (0.3-2.0x). |
| `value` | FLOAT | `0.5` | Multiplier for speed/volume; integer steps for pitch. |
| `custom_instruction` | STRING | empty | Optional raw instruction override; leave empty to build the trained template from mode + value. |
| generation controls | same as clone | | `inference_cfg` defaults to `1.2` here. |

**Output:** `audio` (`AUDIO`)

</details>


<details>
<summary><strong>6. FireRedTTS3 Whisper Transcribe</strong> - Reference transcript helper</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | AUDIO | required | Reference clip to transcribe. |
| `model` | COMBO | `whisper-large-v3-turbo` | Whisper ASR size; downloads into `ComfyUI/models/audio_encoders/` when missing. |
| `dtype` | COMBO | `auto` | `auto`, `bf16`, `fp32`. |
| `language` | COMBO | `auto` | Clip language; auto-detects. |
| `task` | COMBO | `transcribe` | `transcribe` keeps the language; `translate` outputs English. |
| `chunk_length_s` | INT | `30` | Chunk length for longer clips. |
| `download_if_missing` | BOOLEAN | `True` | Download the Whisper model if needed. |

**Outputs:** `transcript` (`STRING`) - connect it to Voice Clone's `prompt_text` (right-click the clone node -> convert `prompt_text` to input); `audio` (`AUDIO`) - the input passed through unchanged, for chaining to the clone node.

</details>

## Usage Notes

- A correct `prompt_text` transcript materially improves cloning; the Whisper node produces it in one click.
- For best cloning, use a prompt clip in the target language/dialect - the output inherits the reference's speaking style. Reference clips longer than ~655s are rejected (RedAE encoder limit); 5-20s clones best.
- Official defaults are preserved: 10 flow steps, CFG 2.0 (clone) / 1.2 (design and edits), stop threshold 0.5, ~64s per-sentence cap, 50 ms cross-fade.
- The upstream LLM-API text normalizer is intentionally excluded; this nodepack makes no external API calls.

## Troubleshooting

- **`flash_attention` / `sageattention` errors** - the packages are optional; use `auto` (sdpa) unless you installed them.
- **Auto language detection weak for a language** - make sure `fasttext-predict` (or `fasttext`) is installed and `lid.176.ftz` was downloaded; otherwise pick the language explicitly.
- **Voice Design/Edit nodes raise "requires the Instruct model"** - load `fireredtts3_instruct` in the loader.
- **Out of memory** - use the bf16 mirror repo and enable ComfyUI DynamicVRAM (AIMDO); the codec nodes and Whisper also page through the same management.

## Credits

- [FireRedTeam/FireRedTTS3](https://github.com/FireRedTeam/FireRedTTS3) - model and original implementation (Apache-2.0)
- [Qwen3](https://github.com/QwenLM/Qwen3), [DiTAR](https://arxiv.org/abs/2502.03930), [X-Codec](https://github.com/zhenye234/xcodec), [CAM++](https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k), [fastText](https://fasttext.cc/), [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)
- bf16 mirror: [drbaph/FireRedTTS3-bf16](https://huggingface.co/drbaph/FireRedTTS3-bf16), int8 ConvRot mirror: [drbaph/FireRedTTS3-int8](https://huggingface.co/drbaph/FireRedTTS3-int8)

Voice cloning is intended for research use. Do not clone voices without consent, and do not use generated audio for illegal activities.
