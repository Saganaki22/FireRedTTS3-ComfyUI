# FireRedTTS3-ComfyUI

**[English](./README.md)** | **中文**

**版本：v0.2.4**

[FireRedTeam/FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3) 的 ComfyUI 节点：24 种语言 + 21 种中文方言的零样本语音克隆、指令驱动的音色设计、语义 + 声学语音编辑、Whisper 参考音频转写，以及 ComfyUI/AIMDO DynamicVRAM 动态显存支持。

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FireRedTeam%2FFireRedTTS3-blue)](https://huggingface.co/FireRedTeam/FireRedTTS3)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-drbaph%2FFireRedTTS3--bf16-green)](https://huggingface.co/drbaph/FireRedTTS3-bf16)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-drbaph%2FFireRedTTS3--int8-green)](https://huggingface.co/drbaph/FireRedTTS3-int8)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Saganaki22/FireRedTTS3-ComfyUI/blob/main/LICENSE)

> 许可说明：FireRedTTS3 由 FireRed 团队以 Apache-2.0 许可发布，仅供学术研究用途。未经授权请勿克隆他人声音。

## 功能

- **原生进程内推理** - Qwen3 主干、PatchEncoder、DiT 流匹配头、RedAE 编解码器与 CAM++ 说话人编码器直接在 ComfyUI 内运行；无外部服务、无远程代码。
- **双模型变体** - `fireredtts3_base`（带语言标签的零样本克隆）与 `fireredtts3_instruct`（克隆 + 音色设计 + 语音编辑，默认），共享同一 RedAE 编解码器。
- **24 种语言 + 21 种中文方言** - 显式语言/方言下拉选择，或 `auto` 使用 FastText lid.176 自动检测（无 fasttext 时回退到中/日/英启发式判断）。
- **音色设计** - 用自然语言描述音色（性别、年龄、音质、语速、口音）直接合成；模型生成的语音属性规划（CoT）以文本形式返回。
- **语音编辑** - 语义编辑（插入/删除/替换词语，返回改写文本）与声学编辑（语速 0.5-2.0 倍、音高 ±6 步、音量 0.3-2.0 倍），使用官方训练模板。
- **内置文本前端** - 本地 wetext 中英文正则化（数字、日期、单位转口语），自动分句 + 交叉淡入淡出拼接。不调用任何外部 API。
- **ComfyUI AUDIO 输入输出** - 参考音频、编辑输入与生成音频均使用标准 ComfyUI `AUDIO`。
- **流步进优化** - 时间步嵌入、逐块 AdaLN 调制参数与旋转频率在每次生成时只计算一次（而非每个 patch 都重算），并预分配生成缓冲：速度提升约 11-16%，输出逐位一致（同种子 A/B 验证）。
- **AIMDO DynamicVRAM 支持** - 核心模型、编解码器、说话人编码器分别注册为 ComfyUI 模型，权重可转换分页，DynamicVRAM 激活时自动接管。
- **bf16 镜像** - 可选的半体积权重，与官方混合精度一致：主干 LLM 与 RedAE 编码器存为 bf16（官方推理本就在 bf16 autocast 下计算），流匹配头与 RedAE 解码器保持 fp32。同种子测试输出与 fp32 版本一致（SNR > 80 dB）。
- **INT8 ConvRot 权重（实验性）** - `FireRedTTS3-int8` 镜像内置已量化的核心 transformer 线性层（Comfy INT8 ConvRot：格式 `int8_tensorwise`、逐行 fp32 缩放、离线 Hadamard 权重旋转、group size 256，使用官方 comfy-kitchen 量化器）。加载器自动识别 `*.comfy_quant` 键并通过 `comfy_kitchen.int8_linear(convrot=True)` 在线旋转激活执行；其余层（嵌入、边界投影、stop_head、Conv1d、RedAE、CAM++）保持浮点。321/332 线性层 → 检查点 7.90 → 3.07 GiB，输出质量已验证。
- **Whisper 转写节点** - 一键把参考音频转成显著提升克隆质量的 `prompt_text` 转写文本，并原样透传输入音频。
- **无需 keep-loaded 开关、无需卸载节点** - 加载器内部自动处理模型切换清理。

## 安装

### 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/FireRedTTS3-ComfyUI.git
cd FireRedTTS3-ComfyUI
python install.py
```

Windows 本地环境：

```powershell
...\venv\Scripts\python.exe ...\ComfyUI\custom_nodes\FireRedTTS3-ComfyUI\install.py
```

安装或更新后请重启 ComfyUI。

`install.py` 同时兼容 **pip** 和 **uv**，且绝不安装、升级或删除 `torch`、`torchaudio`、`transformers`、`numpy` —— 只补装不含 torch 依赖的轻量包（`huggingface_hub`、`safetensors`、`tokenizers`、`accelerate`、`regex`、`tqdm`，以及可选的 `wetext` 和 `fasttext-predict`）。

## Transformers 兼容性

本节点包基于 Transformers **5.3.0+** 构建（在 5.3.0 上测试通过）。它原生加载 FireRedTTS3 而不依赖远程代码 `AutoModel`：直接构建 Qwen3 主干、PatchEncoder、DiT 流匹配头与 RedAE 模块，并按原始键名映射 safetensors 权重。

## 各功能语言支持

| 功能 | 语言支持 |
| --- | --- |
| **音色克隆（base）** | 24 种语言 + 21 种中文方言，可显式选择标签或使用 `auto`（FastText 自动检测） |
| **音色克隆（instruct）** | 从参考音频继承语言；语言下拉框会被忽略（会记录警告日志）。需要带标签的多语言克隆请使用 base 模型。 |
| **声音设计（instruct）** | 仅中文和英文——上游模型的限制：没有参考音频也没有语言标签，模型没有任何语言条件可依（其官方基准也只评测中英）。其他语言请先用中/英文设计声音，再用 base 模型加语言标签克隆该输出。 |
| **语义 / 声学编辑（instruct）** | 继承输入音频的语言（已验证：希腊语音频 1.5 倍速编辑后希腊语内容保持不变）。 |

## 模型文件

权重按来源存放在 `ComfyUI/models/fireredtts3/` 下（主文件夹优先；同时搜索 extra_model_paths.yaml 额外路径与符号链接）。缺失文件在 `download_if_missing` 开启时自动下载，否则报错并提示放置路径。

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 fireredtts3/
        ├── 📂 drbaph_FireRedTTS3-bf16/        (bf16 镜像，默认)
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
        ├── 📂 drbaph_FireRedTTS3-int8/        (int8 ConvRot 镜像，布局相同)
        ├── 📂 FireRedTeam_FireRedTTS3/        (官方 fp32，布局相同)
        └── 📂 fasttext/
            └── lid.176.ftz                    (共享语言识别模型)
```

只下载所选变体；两个变体共享 `redae/`、`campp/` 与 `text_tokenizer/`。

| 组件 | 官方 fp32 | bf16 镜像 | int8 ConvRot 镜像 |
| --- | --- | --- | --- |
| `fireredtts3_base` | 8.48 GB | 4.70 GiB | 3.30 GB |
| `fireredtts3_instruct` | 8.48 GB | 4.69 GiB | 3.30 GB |
| `redae` | 3.78 GB | 2.46 GiB | 原样保留 |
| `campp` + 分词器 + fasttext | 约 45 MB | 约 45 MB | 约 45 MB |

手动安装：把文件放入 `ComfyUI/models/fireredtts3/drbaph_FireRedTTS3-bf16/`（或对应仓库文件夹），加载器会直接使用，无需下载。

## 节点

<details>
<summary><strong>1. FireRedTTS3 Load Model（加载模型）</strong></summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `repo` | COMBO | `FireRedTTS3-bf16` | 权重来源：`FireRedTTS3-bf16`（推荐）、`FireRedTTS3-int8`（最小、实验性）、`FireRedTTS3-fp32`（官方）。缺失文件在 `download_if_missing` 开启时下载，否则报错并提示放置路径。 |
| `variant` | COMBO | `fireredtts3_instruct` | `fireredtts3_base`（克隆 + 语言标签）或 `fireredtts3_instruct`（克隆 + 设计 + 编辑）。 |
| `dtype` | COMBO | `auto` | `auto`、`bf16`、`fp32`。bf16 为主干 LLM + RedAE 编码器存 bf16，流匹配头/解码器保持 fp32（官方混合精度）。 |
| `device` | COMBO | `auto` | `auto`、`cuda`、`cpu`。`auto` 跟随 ComfyUI 当前设备。 |
| `attention` | COMBO | `auto` | 已安装 flash_attn 且兼容（CUDA + bf16 计算）时 `auto` 使用 `flash_attention`，否则 `sdpa`；也可显式选择 `sdpa` / `flash_attention` / `sageattention`。fp32 的 RedAE 解码器始终使用 sdpa。 |
| `download_if_missing` | BOOLEAN | `True` | 缺失时下载所选权重、编解码器、分词器、CAM++ 与 FastText 文件。 |

**输出：** `firered_model` (`FIREREDTTS3_MODEL`)

</details>

<details>
<summary><strong>2. FireRedTTS3 Voice Clone（语音克隆）</strong> - 零样本克隆（Base 或 Instruct）</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `firered_model` | FIREREDTTS3_MODEL | 必填 | 加载器输出。 |
| `text` | STRING | 示例文本 | 用参考音色合成的文本。 |
| `prompt_audio` | AUDIO | 必填 | 干净的参考音频片段。 |
| `prompt_text` | STRING | 空 | 参考音频的精确转写。强烈建议提供；右键节点可转换为输入并连接 Whisper。 |
| `language` | COMBO | `auto` | 24 种语言 + 21 种方言，或自动检测。与参考音频语言一致时克隆效果最佳。 |
| `n_timesteps` | INT | `10` | 每个音频 patch 的流匹配步数（官方默认）。 |
| `inference_cfg` | FLOAT | `2.0` | 流头 CFG 强度。`0` 关闭。 |
| `stop_threshold` | FLOAT | `0.5` | 停止 token 概率阈值。 |
| `seed` | INT | `42` | `0` 随机；正值可复现。 |
| `max_audio_seconds` | FLOAT | `64.0` | 每句时长上限（64 秒为官方上限）。 |
| `do_tn` | BOOLEAN | `True` | 文本正则化（中英文 wetext）。 |
| `do_split` | BOOLEAN | `True` | 长文本自动分句并交叉淡入淡出拼接。 |
| `cross_fade_ms` | FLOAT | `50.0` | 句间交叉淡入淡出时长。 |

**输出：** `audio` (`AUDIO`)

</details>

<details>
<summary><strong>3. FireRedTTS3 Voice Design（音色设计）</strong> - 按描述生成音色（Instruct）</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `firered_model` | FIREREDTTS3_MODEL | 必填 | Instruct 模型。 |
| `instruction` | STRING | 示例 | 音色描述：性别、年龄、音质、情绪、语速、口音。 |
| `text` | STRING | 示例文本 | 用设计音色朗读的文本。 |
| `language` | COMBO | `auto` | 仅用于文本正则化。 |
| `text_temperature` | FLOAT | `0.7` | 语音属性规划文本采样温度。 |
| `text_top_p` / `text_top_k` / `text_repetition_penalty` | FLOAT/INT/FLOAT | `0.8` / `20` / `1.0` | 规划文本采样参数。 |
| 生成控制 | 同克隆 | | 此处 `inference_cfg` 默认 `1.2`。 |

**输出：** `audio` (`AUDIO`)、`voice_plan` (`STRING`) - 模型的语音属性规划。

</details>

<details>
<summary><strong>4. FireRedTTS3 Semantic Edit（语义编辑）</strong> - 插入/删除/替换词语（Instruct）</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `firered_model` | FIREREDTTS3_MODEL | 必填 | Instruct 模型。 |
| `audio` | AUDIO | 必填 | 待编辑的语音。 |
| `instruction` | STRING | 示例 | 例如 `Replace 'cats' with 'dogs'.` |
| 生成控制 | 同克隆 | | 此处 `inference_cfg` 默认 `1.2`。 |

**输出：** `audio` (`AUDIO`)、`edited_text` (`STRING`) - 模型改写后的文本。

</details>

<details>
<summary><strong>5. FireRedTTS3 Acoustic Edit（声学编辑）</strong> - 语速/音高/音量（Instruct）</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `firered_model` | FIREREDTTS3_MODEL | 必填 | Instruct 模型。 |
| `audio` | AUDIO | 必填 | 待变换的语音。 |
| `mode` | COMBO | `speed` | `speed`（0.5-2.0 倍）、`pitch`（±6 步）、`volume`（0.3-2.0 倍）。 |
| `value` | FLOAT | `0.5` | speed/volume 为倍率；pitch 为整数步。 |
| `custom_instruction` | STRING | 空 | 可选的原始指令覆盖；留空则按 mode + value 生成官方模板。 |
| 生成控制 | 同克隆 | | 此处 `inference_cfg` 默认 `1.2`。 |

**输出：** `audio` (`AUDIO`)

</details>

<details>
<summary><strong>6. FireRedTTS3 Whisper Transcribe（转写）</strong> - 参考转写助手</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `audio` | AUDIO | 必填 | 待转写的参考片段。 |
| `model` | COMBO | `whisper-large-v3-turbo` | Whisper 型号；缺失时下载到 `ComfyUI/models/audio_encoders/`。 |
| `dtype` | COMBO | `auto` | `auto`、`bf16`、`fp32`。 |
| `language` | COMBO | `auto` | 片段语言；自动检测。 |
| `task` | COMBO | `transcribe` | `transcribe` 保留原语言；`translate` 输出英语。 |
| `chunk_length_s` | INT | `30` | 长片段分块长度。 |
| `download_if_missing` | BOOLEAN | `True` | 需要时下载 Whisper 模型。 |

**输出：** `audio` (`AUDIO`) - 原样透传输入音频，便于串接到克隆节点；`transcript` (`STRING`) - 连接到 Voice Clone 的 `prompt_text`（右键克隆节点 -> 转换为输入）。

</details>

## 使用说明

- 正确的 `prompt_text` 转写会显著提升克隆质量；Whisper 节点一键生成。
- 克隆时尽量使用目标语言/方言的参考片段 —— 输出会继承参考的说话风格（跨语言克隆效果较弱，属官方模型特性）。超过约 655 秒的参考音频会被拒绝（RedAE 编码器上限）；5-20 秒克隆效果最佳。
- 参考音频质量不佳时优先使用 instruct 变体（对嘈杂参考更鲁棒）；干净参考 + 显式语言标签时 base 变体表现出色。
- 保留官方默认参数：流匹配 10 步、CFG 2.0（克隆）/ 1.2（设计与编辑）、停止阈值 0.5、每句约 64 秒上限、50 ms 交叉淡入淡出。
- 上游的 LLM-API 文本正则化被有意排除；本节点包不发起任何外部 API 调用。

## 故障排查

- **`flash_attention` / `sageattention` 报错** - 这些是可选包；未安装时 `auto` 自动使用 sdpa。
- **dtype=fp32 时 flash_attention 回退** - flash_attn 只接受 bf16/fp16 输入，fp32 模式会自动回退 sdpa 并打印警告。
- **某语言自动检测不准** - 确认已安装 `fasttext-predict`（或 `fasttext`）且 `lid.176.ftz` 已下载；否则请显式选择语言。
- **Voice Design / Edit 节点报 "needs the Instruct model"** - 在加载器中把 variant 改为 `fireredtts3_instruct`。
- **显存不足** - 使用 `FireRedTTS3-int8` 或 bf16 镜像并启用 ComfyUI DynamicVRAM（AIMDO）；Whisper 也走同一内存管理。

## 致谢

- [FireRedTeam/FireRedTTS3](https://github.com/FireRedTeam/FireRedTTS3) - 模型与原始实现（Apache-2.0）
- [Qwen3](https://github.com/QwenLM/Qwen3)、[DiTAR](https://arxiv.org/abs/2502.03930)、[X-Codec](https://github.com/zhenye234/xcodec)、[CAM++](https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k)、[fastText](https://fasttext.cc/)、[WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)
- bf16 镜像：[drbaph/FireRedTTS3-bf16](https://huggingface.co/drbaph/FireRedTTS3-bf16)，int8 ConvRot 镜像：[drbaph/FireRedTTS3-int8](https://huggingface.co/drbaph/FireRedTTS3-int8)

语音克隆仅供研究用途。未经授权请勿克隆他人声音，请勿将生成音频用于任何违法活动。
